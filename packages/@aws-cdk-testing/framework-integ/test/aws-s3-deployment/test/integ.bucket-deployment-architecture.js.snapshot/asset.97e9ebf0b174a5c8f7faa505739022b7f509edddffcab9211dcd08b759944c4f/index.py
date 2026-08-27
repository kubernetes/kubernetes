import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.parse
from urllib.request import Request, urlopen
from uuid import uuid4
from zipfile import ZipFile

import boto3
from botocore.config import Config
from botocore.exceptions import WaiterError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cloudfront = boto3.client('cloudfront', config=Config(
    retries = {
        'max_attempts': 10,
        'mode': 'standard',
    }
))
s3 = boto3.client('s3')

CFN_SUCCESS = "SUCCESS"
CFN_FAILED = "FAILED"
ENV_KEY_MOUNT_PATH = "MOUNT_PATH"
ENV_KEY_SKIP_CLEANUP = "SKIP_CLEANUP"

AWS_CLI_CONFIG_FILE = "/tmp/aws_cli_config"
CUSTOM_RESOURCE_OWNER_TAG = "aws-cdk:cr-owned"

os.putenv('AWS_CONFIG_FILE', AWS_CLI_CONFIG_FILE)

def handler(event, context):

    def cfn_error(message=None):
        if message:
            logger.error("| cfn_error: %s" % message.encode())
        cfn_send(event, context, CFN_FAILED, reason=message, physicalResourceId=event.get('PhysicalResourceId', None))


    try:
        # We are not logging ResponseURL as this is a pre-signed S3 URL, and could be used to tamper
        # with the response CloudFormation sees from this Custom Resource execution.
        logger.info({ key:value for (key, value) in event.items() if key != 'ResponseURL'})

        # cloudformation request type (create/update/delete)
        request_type = event['RequestType']

        # extract resource properties
        props = event['ResourceProperties']
        old_props = event.get('OldResourceProperties', {})
        physical_id = event.get('PhysicalResourceId', None)

        try:
            source_bucket_names = props['SourceBucketNames']
            source_object_keys  = props['SourceObjectKeys']
            source_markers      = props.get('SourceMarkers', None)
            source_markers_config = props.get('SourceMarkersConfig', None)
            dest_bucket_name    = props['DestinationBucketName']
            dest_bucket_prefix  = props.get('DestinationBucketKeyPrefix', '')
            extract             = props.get('Extract', 'true') == 'true'
            retain_on_delete    = props.get('RetainOnDelete', "true") == "true"
            distribution_id     = props.get('DistributionId', '')
            wait_for_distribution_invalidation = props.get('WaitForDistributionInvalidation', True)
            user_metadata       = props.get('UserMetadata', {})
            system_metadata     = props.get('SystemMetadata', {})
            prune               = props.get('Prune', 'true').lower() == 'true'
            exclude             = props.get('Exclude', [])
            include             = props.get('Include', [])
            sign_content        = props.get('SignContent', 'false').lower() == 'true'
            output_object_keys  = props.get('OutputObjectKeys', 'true') == 'true'

            # backwards compatibility - if "SourceMarkers" is not specified,
            # assume all sources have an empty market map
            if source_markers is None:
                source_markers = [{} for i in range(len(source_bucket_names))]
            if source_markers_config is None:
                source_markers_config = [{} for i in range(len(source_bucket_names))]

            default_distribution_path = dest_bucket_prefix
            if not default_distribution_path.endswith("/"):
                default_distribution_path += "/"
            if not default_distribution_path.startswith("/"):
                default_distribution_path = "/" + default_distribution_path
            default_distribution_path += "*"

            distribution_paths = props.get('DistributionPaths', [default_distribution_path])
        except KeyError as e:
            cfn_error("missing request resource property %s. props: %s" % (str(e), props))
            return

        # configure aws cli options after resetting back to the defaults for each request
        if os.path.exists(AWS_CLI_CONFIG_FILE):
                os.remove(AWS_CLI_CONFIG_FILE)
        if sign_content:
                aws_command("configure", "set", "default.s3.payload_signing_enabled", "true")

        # treat "/" as if no prefix was specified
        if dest_bucket_prefix == "/":
            dest_bucket_prefix = ""

        s3_source_zips = list(map(lambda name, key: "s3://%s/%s" % (name, key), source_bucket_names, source_object_keys))
        s3_dest = "s3://%s/%s" % (dest_bucket_name, dest_bucket_prefix)
        old_s3_dest = "s3://%s/%s" % (old_props.get("DestinationBucketName", ""), old_props.get("DestinationBucketKeyPrefix", ""))


        # obviously this is not
        if old_s3_dest == "s3:///":
            old_s3_dest = None

        logger.info("| s3_dest: %s" % sanitize_message(s3_dest))
        logger.info("| old_s3_dest: %s" % sanitize_message(old_s3_dest))

        # if we are creating a new resource, allocate a physical id for it
        # otherwise, we expect physical id to be relayed by cloudformation
        if request_type == "Create":
            physical_id = "aws.cdk.s3deployment.%s" % str(uuid4())
        else:
            if not physical_id:
                cfn_error("invalid request: request type is '%s' but 'PhysicalResourceId' is not defined" % request_type)
                return

        # delete or create/update (only if "retain_on_delete" is false)
        if request_type == "Delete" and not retain_on_delete:
            if not bucket_owned(dest_bucket_name, dest_bucket_prefix):
                aws_command("s3", "rm", s3_dest, "--recursive")

        # if we are updating without retention and the destination changed, delete first
        if request_type == "Update" and not retain_on_delete and old_s3_dest != s3_dest:
            if not old_s3_dest:
                logger.warn("cannot delete old resource without old resource properties")
                return

            aws_command("s3", "rm", old_s3_dest, "--recursive")

        if request_type == "Update" or request_type == "Create":
            s3_deploy(s3_source_zips, s3_dest, user_metadata, system_metadata, prune, exclude, include, source_markers, extract, source_markers_config)

        if distribution_id:
            cloudfront_invalidate(distribution_id, distribution_paths, wait_for_distribution_invalidation)

        cfn_send(event, context, CFN_SUCCESS, physicalResourceId=physical_id, responseData={
            # Passing through the ARN sequences dependencees on the deployment
            'DestinationBucketArn': props.get('DestinationBucketArn'),
            **({'SourceObjectKeys': props.get('SourceObjectKeys')} if output_object_keys else {'SourceObjectKeys': []})
        })
    except KeyError as e:
        cfn_error("invalid request. Missing key %s" % str(e))
    except Exception as e:
        logger.exception(e)
        cfn_error(str(e))

#---------------------------------------------------------------------------------------------------
# Sanitize the message to mitigate CWE-117 and CWE-93 vulnerabilities
def sanitize_message(message):
    if not message:
        return message

    # Sanitize the message to prevent log injection and HTTP response splitting
    sanitized_message = message.replace('\n', '').replace('\r', '')

    # Encode the message to handle special characters
    encoded_message = urllib.parse.quote(sanitized_message)

    return encoded_message

#---------------------------------------------------------------------------------------------------
# populate all files from s3_source_zips to a destination bucket
def s3_deploy(s3_source_zips, s3_dest, user_metadata, system_metadata, prune, exclude, include, source_markers, extract, source_markers_config):
    # list lengths are equal
    if len(s3_source_zips) != len(source_markers):
        raise Exception("'source_markers' and 's3_source_zips' must be the same length")

    # create a temporary working directory in /tmp or if enabled an attached efs volume
    if ENV_KEY_MOUNT_PATH in os.environ:
        workdir = os.getenv(ENV_KEY_MOUNT_PATH) + "/" + str(uuid4())
        os.mkdir(workdir)
    else:
        workdir = tempfile.mkdtemp()

    logger.info("| workdir: %s" % workdir)

    # create a directory into which we extract the contents of the zip file
    contents_dir=os.path.join(workdir, 'contents')
    os.mkdir(contents_dir)

    try:
        # download the archive from the source and extract to "contents"
        for i in range(len(s3_source_zips)):
            s3_source_zip = s3_source_zips[i]
            markers       = source_markers[i]
            markers_config = source_markers_config[i]

            if extract:
                archive=os.path.join(workdir, str(uuid4()))
                logger.info("archive: %s" % archive)
                aws_command("s3", "cp", s3_source_zip, archive)
                logger.info("| extracting archive to: %s\n" % contents_dir)
                logger.info("| markers: %s" % markers)
                extract_and_replace_markers(archive, contents_dir, markers, markers_config)
            else:
                logger.info("| copying archive to: %s\n" % contents_dir)
                aws_command("s3", "cp", s3_source_zip, contents_dir)

        # sync from "contents" to destination

        s3_command = ["s3", "sync"]

        if prune:
          s3_command.append("--delete")

        if exclude:
          for filter in exclude:
            s3_command.extend(["--exclude", filter])

        if include:
          for filter in include:
            s3_command.extend(["--include", filter])

        s3_command.extend([contents_dir, s3_dest])
        s3_command.extend(create_metadata_args(user_metadata, system_metadata))
        aws_command(*s3_command)
    finally:
        if not os.getenv(ENV_KEY_SKIP_CLEANUP):
            shutil.rmtree(workdir)

#---------------------------------------------------------------------------------------------------
# invalidate files in the CloudFront distribution edge caches
def cloudfront_invalidate(distribution_id, distribution_paths, wait_for_invalidation):
    invalidation_resp = cloudfront.create_invalidation(
        DistributionId=distribution_id,
        InvalidationBatch={
            'Paths': {
                'Quantity': len(distribution_paths),
                'Items': distribution_paths
            },
            'CallerReference': str(uuid4()),
        })
    if wait_for_invalidation:
        try:
            # Wait for a maximum of 13 minutes for invalidation to complete.
            cloudfront.get_waiter('invalidation_completed').wait(
                DistributionId=distribution_id,
                Id=invalidation_resp['Invalidation']['Id'],
                WaiterConfig={
                    'Delay': 20,
                    'MaxAttempts': (13*60)//20,
                }
            )
        except WaiterError as e:
            raise RuntimeError(f"Unable to confirm that cache invalidation was successful. This may be a CloudFront regression as reported in https://github.com/aws/aws-cdk/issues/15891") from e


#---------------------------------------------------------------------------------------------------
# set metadata
def create_metadata_args(raw_user_metadata, raw_system_metadata):
    if len(raw_user_metadata) == 0 and len(raw_system_metadata) == 0:
        return []

    format_system_metadata_key = lambda k: k.lower()
    format_user_metadata_key = lambda k: k.lower()

    system_metadata = { format_system_metadata_key(k): v for k, v in raw_system_metadata.items() }
    user_metadata = { format_user_metadata_key(k): v for k, v in raw_user_metadata.items() }

    flatten = lambda l: [item for sublist in l for item in sublist]
    system_args = flatten([[f"--{k}", v] for k, v in system_metadata.items()])
    user_args = ["--metadata", json.dumps(user_metadata, separators=(',', ':'))] if len(user_metadata) > 0 else []

    return system_args + user_args + ["--metadata-directive", "REPLACE"]

#---------------------------------------------------------------------------------------------------
# executes an "aws" cli command
def aws_command(*args):
    aws="/opt/awscli/aws" # from AwsCliLayer
    logger.info("| aws %s" % ' '.join(args))
    subprocess.check_call([aws] + list(args))

#---------------------------------------------------------------------------------------------------
# sends a response to cloudformation
def cfn_send(event, context, responseStatus, responseData={}, physicalResourceId=None, noEcho=False, reason=None):

    responseUrl = event['ResponseURL']

    responseBody = {}
    responseBody['Status'] = responseStatus
    responseBody['Reason'] = reason or ('See the details in CloudWatch Log Stream: ' + context.log_stream_name)
    responseBody['PhysicalResourceId'] = physicalResourceId or context.log_stream_name
    responseBody['StackId'] = event['StackId']
    responseBody['RequestId'] = event['RequestId']
    responseBody['LogicalResourceId'] = event['LogicalResourceId']
    responseBody['NoEcho'] = noEcho
    responseBody['Data'] = responseData

    body = json.dumps(responseBody)
    logger.info("| response body:\n" + body)

    headers = {
        'content-type' : '',
        'content-length' : str(len(body))
    }

    try:
        request = Request(responseUrl, method='PUT', data=bytes(body.encode('utf-8')), headers=headers)
        with contextlib.closing(urlopen(request)) as response:
            logger.info("| status code: " + response.reason)
    except Exception as e:
        logger.error("| unable to send response to CloudFormation")
        logger.exception(e)


#---------------------------------------------------------------------------------------------------
# check if bucket is owned by a custom resource
# if it is then we don't want to delete content
def bucket_owned(bucketName, keyPrefix):
    tag = CUSTOM_RESOURCE_OWNER_TAG
    if keyPrefix != "":
        tag = tag + ':' + keyPrefix
    try:
        request = s3.get_bucket_tagging(
            Bucket=bucketName,
        )
        return any((x["Key"].startswith(tag)) for x in request["TagSet"])
    except Exception as e:
        logger.info("| error getting tags from bucket")
        logger.exception(e)
        return False

# extract archive and replace markers in output files
def extract_and_replace_markers(archive, contents_dir, markers, markers_config):
    with ZipFile(archive, "r") as zip:
        zip.extractall(contents_dir)

        # replace markers for this source
        for file in zip.namelist():
            file_path = os.path.join(contents_dir, file)
            if os.path.isdir(file_path): continue
            replace_markers(file_path, markers, markers_config)

def prepare_json_safe_markers(markers):
    """Pre-process markers to ensure JSON-safe values"""
    safe_markers = {}
    for key, value in markers.items():
        # Serialize the value as JSON to handle escaping if the value is a string
        serialized = json.dumps(value)
        if serialized.startswith('"') and serialized.endswith('"'):
            json_safe_value = json.dumps(value)[1:-1]  # Remove surrounding quotes
        else:
            json_safe_value = serialized
        safe_markers[key.encode('utf-8')] = json_safe_value.encode('utf-8')
    return safe_markers

def replace_markers(filename, markers, markers_config):
    """Replace markers in a file, with special handling for JSON files."""
    # if there are no markers, skip
    if not markers:
        return
    
    outfile = filename + '.new'
    json_escape = markers_config.get('jsonEscape', 'false').lower()
    if json_escape == 'true':
        replace_tokens = prepare_json_safe_markers(markers)
    else:
        replace_tokens = dict([(k.encode('utf-8'), v.encode('utf-8')) for k, v in markers.items()])

    # Handle content with line-by-line binary replacement
    with open(filename, 'rb') as fi, open(outfile, 'wb') as fo:
        # Process line by line to handle large files
        for line in fi:
            for token, replacement in replace_tokens.items():
                line = line.replace(token, replacement)
            fo.write(line)

    # Delete the original file and rename the new one to the original
    os.remove(filename)
    os.rename(outfile, filename)

def replace_markers_in_json(json_object, replace_tokens):
    """Replace markers in JSON content with proper escaping."""
    try:
        def replace_in_structure(obj):
            if isinstance(obj, str):
                # Convert string to bytes for consistent replacement
                result = obj.encode('utf-8')
                for token, replacement in replace_tokens.items():
                    result = result.replace(token, replacement)
                # Convert back to string
                return result.decode('utf-8')
            elif isinstance(obj, dict):
                return {k: replace_in_structure(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_structure(item) for item in obj]
            return obj

        # Process the whole structure
        processed = replace_in_structure(json_object)
        return json.dumps(processed)
    except Exception as e:
        logger.error(f'Error processing JSON: {e}')
        logger.exception(e)
        return json_object
