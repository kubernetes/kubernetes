import json
import logging
import os
import subprocess

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# these are coming from the kubectl layer
os.environ['PATH'] = '/opt/kubectl:/opt/awscli:' + os.environ['PATH']

outdir = os.environ.get('TEST_OUTDIR', '/tmp')
kubeconfig = os.path.join(outdir, 'kubeconfig')


def apply_handler(event, context):
    logger.info(json.dumps(dict(event, ResponseURL='...')))

    request_type = event['RequestType']
    props = event['ResourceProperties']

    # resource properties (all required)
    cluster_name  = props['ClusterName']
    manifest_text = props['Manifest']
    role_arn      = props['RoleArn']
    prune_label   = props.get('PruneLabel', None)
    overwrite     = props.get('Overwrite', 'false').lower() == 'true'
    skip_validation = props.get('SkipValidation', 'false').lower() == 'true'

    # "log in" to the cluster
    cmd = [ 'aws', 'eks', 'update-kubeconfig',
        '--role-arn', role_arn,
        '--name', cluster_name,
        '--kubeconfig', kubeconfig
    ]
    logger.info(f'Running command: {cmd}')
    subprocess.check_call(cmd)

    if os.path.isfile(kubeconfig):
        os.chmod(kubeconfig, 0o600)

    # write resource manifests in sequence: { r1 }{ r2 }{ r3 } (this is how
    # a stream of JSON objects can be included in a k8s manifest).
    manifest_list = json.loads(manifest_text)
    manifest_file = os.path.join(outdir, 'manifest.yaml')
    with open(manifest_file, "w") as f:
        f.writelines(map(lambda obj: json.dumps(obj), manifest_list))

    logger.info("manifest written to: %s" % manifest_file)

    kubectl_opts = []
    if skip_validation:
        kubectl_opts.extend(['--validate=false'])

    if request_type == 'Create':
        # if "overwrite" is enabled, then we use "apply" for CREATE operations
        # which technically means we can determine the desired state of an
        # existing resource.
        if overwrite:
            kubectl('apply', manifest_file, *kubectl_opts)
        else:
            # --save-config will allow us to use "apply" later
            kubectl_opts.extend(['--save-config'])
            kubectl('create', manifest_file, *kubectl_opts)
    elif request_type == 'Update':
        if prune_label is not None:
            kubectl_opts.extend(['--prune', '-l', prune_label])

        kubectl('apply', manifest_file, *kubectl_opts)
    elif request_type == "Delete":
        try:
            kubectl('delete', manifest_file)
        except Exception as e:
            logger.info("delete error: %s" % e)


def kubectl(verb, file, *opts):
    maxAttempts = 3
    retry = maxAttempts
    while retry > 0:
        try:
            cmd = ['kubectl', verb, '--kubeconfig', kubeconfig, '-f', file] + list(opts)
            logger.info(f'Running command: {cmd}')
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            output = exc.output
            if b'i/o timeout' in output and retry > 0:
                retry = retry - 1
                logger.info("kubectl timed out, retries left: %s" % retry)
            else:
                raise Exception(output)
        else:
            logger.info(output)
            return
    raise Exception(f'Operation failed after {maxAttempts} attempts: {output}')
