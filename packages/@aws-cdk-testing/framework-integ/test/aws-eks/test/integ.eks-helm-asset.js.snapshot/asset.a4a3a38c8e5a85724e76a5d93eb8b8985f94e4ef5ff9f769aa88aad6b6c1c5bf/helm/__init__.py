import json
import logging
import os
import re
import subprocess
import shutil
import tempfile
import zipfile
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# these are coming from the kubectl layer
os.environ['PATH'] = '/opt/helm:/opt/awscli:' + os.environ['PATH']

outdir = os.environ.get('TEST_OUTDIR', '/tmp')
kubeconfig = os.path.join(outdir, 'kubeconfig')

def get_chart_asset_from_url(chart_asset_url):
    chart_zip = os.path.join(outdir, 'chart.zip')
    shutil.rmtree(chart_zip, ignore_errors=True)
    subprocess.check_call(['aws', 's3', 'cp', chart_asset_url, chart_zip])
    chart_dir = os.path.join(outdir, 'chart')
    shutil.rmtree(chart_dir, ignore_errors=True)
    os.mkdir(chart_dir)
    with zipfile.ZipFile(chart_zip, 'r') as zip_ref:
        zip_ref.extractall(chart_dir)
    return chart_dir

def is_ecr_public_available(region):
    s = boto3.Session()
    return s.get_partition_for_region(region) == 'aws'

def helm_handler(event, context):
    logger.info(json.dumps(dict(event, ResponseURL='...')))

    request_type = event['RequestType']
    props = event['ResourceProperties']

    # resource properties
    cluster_name     = props['ClusterName']
    role_arn         = props['RoleArn']
    release          = props['Release']
    chart            = props.get('Chart', None)
    chart_asset_url  = props.get('ChartAssetURL', None)
    version          = props.get('Version', None)
    wait             = props.get('Wait', False)
    atomic           = props.get('Atomic', False)
    timeout          = props.get('Timeout', None)
    namespace        = props.get('Namespace', None)
    create_namespace = props.get('CreateNamespace', None)
    repository       = props.get('Repository', None)
    values_text      = props.get('Values', None)
    skip_crds        = props.get('SkipCrds', False)

    # "log in" to the cluster
    subprocess.check_call([ 'aws', 'eks', 'update-kubeconfig',
        '--role-arn', role_arn,
        '--name', cluster_name,
        '--kubeconfig', kubeconfig
    ])

    if os.path.isfile(kubeconfig):
        os.chmod(kubeconfig, 0o600)

    # Write out the values to a file and include them with the install and upgrade
    values_file = None
    if not request_type == "Delete" and not values_text is None:
        values = json.loads(values_text)
        values_file = os.path.join(outdir, 'values.yaml')
        with open(values_file, "w") as f:
            f.write(json.dumps(values, indent=2))

    if request_type == 'Create' or request_type == 'Update':
        # Ensure chart or chart_asset_url are set
        if chart == None and chart_asset_url == None:
            raise RuntimeError(f'chart or chartAsset must be specified')

        if chart_asset_url != None:
            assert(chart==None)
            assert(repository==None)
            assert(version==None)
            if not chart_asset_url.startswith('s3://'):
              raise RuntimeError(f'ChartAssetURL must point to as s3 location but is {chart_asset_url}')
            # future work: support versions from s3 assets
            chart = get_chart_asset_from_url(chart_asset_url)

        if repository is not None and repository.startswith('oci://'):
            tmpdir = tempfile.TemporaryDirectory()
            chart_dir = get_chart_from_oci(tmpdir.name, repository, version)
            chart = chart_dir
            # Chart is now local — clear repository and version so helm() doesn't
            # pass --repo/--version to "helm upgrade". Helm v4 (kubectl-v35+)
            # rejects --repo with oci:// URLs ("invalid reference"), unlike v3.
            repository = None
            version = None

        helm('upgrade', release, chart, repository, values_file, namespace, version, wait, timeout, create_namespace, skip_crds, atomic=atomic)
    elif request_type == "Delete":
        try:
            helm('uninstall', release, namespace=namespace, wait=wait, timeout=timeout)
        except Exception as e:
            logger.error("Delete error: %s", str(e))


def get_oci_cmd(repository, version):
    # Generates OCI command based on pattern. Public ECR vs Private ECR are treated differently.
    private_ecr_pattern = 'oci://(?P<registry>\d+\.dkr\.ecr\.(?P<region>[a-z0-9\-]+)\.(?P<domain>[a-z0-9\.-]+))*'
    public_ecr_pattern = 'oci://(?P<registry>public\.ecr\.aws)*'

    private_registry = re.match(private_ecr_pattern, repository).groupdict()
    public_registry = re.match(public_ecr_pattern, repository).groupdict()

    # Build helm pull command as array
    helm_cmd = ['helm', 'pull', repository, '--version', version , '--untar']

    if private_registry['registry'] is not None:
        logger.info("Found AWS private repository")
        ecr_login = ['aws', 'ecr', 'get-login-password', '--region', private_registry['region']]
        helm_registry_login = ['helm', 'registry', 'login', '--username', 'AWS', '--password-stdin', private_registry['registry']]
        return {'ecr_login': ecr_login, 'helm_registry_login': helm_registry_login, 'helm': helm_cmd}
    elif public_registry['registry'] is not None:
        logger.info("Found AWS public repository, will use default region as deployment")
        region = os.environ.get('AWS_REGION', 'us-east-1')

        if is_ecr_public_available(region):
            # Public ECR auth is always in us-east-1: https://docs.aws.amazon.com/AmazonECR/latest/public/public-registry-auth.html
            ecr_login = ['aws', 'ecr-public', 'get-login-password', '--region', 'us-east-1']
            helm_registry_login = ['helm', 'registry', 'login', '--username', 'AWS', '--password-stdin', public_registry['registry']]
            return {'ecr_login': ecr_login, 'helm_registry_login': helm_registry_login, 'helm': helm_cmd}
        else:
            # No login required for public ECR in non-aws regions
            # see https://helm.sh/docs/helm/helm_registry_login/
            return {'helm': helm_cmd}
    else:
        logger.error("OCI repository format not recognized, falling back to helm pull")
        return {'helm': helm_cmd}


def get_chart_from_oci(tmpdir, repository=None, version=None):
    from subprocess import Popen, PIPE

    commands = get_oci_cmd(repository, version)

    maxAttempts = 3
    retry = maxAttempts
    while retry > 0:
        try:
            # Execute login commands if needed
            if 'ecr_login' in commands and 'helm_registry_login' in commands:
                logger.info("Running login command: %s", commands['ecr_login'])
                logger.info("Running registry login command: %s", commands['helm_registry_login'])

                # Start first process: aws ecr get-login-password
                # NOTE: We do NOT call p1.wait() here before starting p2.
                # Doing so could deadlock if p1's output fills the pipe buffer
                # before p2 starts reading. Instead, start p2 immediately so it
                # can consume p1's stdout as it's produced.
                p1 = Popen(commands['ecr_login'], stdout=PIPE, stderr=PIPE, cwd=tmpdir)

                # Start second process: helm registry login
                p2 = Popen(commands['helm_registry_login'], stdin=p1.stdout, stdout=PIPE, stderr=PIPE, cwd=tmpdir)
                p1.stdout.close()  # Allow p1 to receive SIGPIPE if p2 exits early

                # Wait for p2 to finish first (ensures full pipeline completes)
                _, p2_err = p2.communicate()

                # Now wait for p1 so we have a complete stderr and an exit code
                p1.wait()

                # Handle p1 failure
                if p1.returncode != 0:
                    p1_err = p1.stderr.read().decode('utf-8', errors='replace') if p1.stderr else ''
                    logger.error(
                        "ECR get-login-password failed for repository %s. Error: %s",
                        repository,
                        p1_err or "No error details"
                    )
                    raise subprocess.CalledProcessError(p1.returncode, commands['ecr_login'], p1_err.encode())

                # Handle p2 failure
                if p2.returncode != 0:
                    p1.kill()
                    logger.error(
                        "Helm registry authentication failed for repository %s. Error: %s",
                        repository,
                        p2_err.decode('utf-8', errors='replace') or "No error details"
                    )
                    raise subprocess.CalledProcessError(p2.returncode, commands['helm_registry_login'], p2_err)

            # Execute helm pull command
            logger.info("Running helm command: %s", commands['helm'])
            output = subprocess.check_output(commands['helm'], stderr=subprocess.STDOUT, cwd=tmpdir)
            logger.info(output.decode('utf-8', errors='replace'))

            # effectively returns "$tmpDir/$lastPartOfOCIUrl", because this is how helm pull saves OCI artifact.
            # Eg. if we have oci://9999999999.dkr.ecr.us-east-1.amazonaws.com/foo/bar/pet-service repository, helm saves artifact under $tmpDir/pet-service
            return os.path.join(tmpdir, repository.rpartition('/')[-1])
        
        except subprocess.CalledProcessError as exc:
            output = exc.output
            if b'Broken pipe' in output:
                retry = retry - 1
                logger.info("Broken pipe, retries left: %s" % retry)
            else:
                error_message = output.decode('utf-8', errors='replace')
                logger.error("OCI command failed: %s", commands['helm'])
                logger.error("Error output: %s", error_message)
                raise Exception(output)
    
    raise Exception(f'Operation failed after {maxAttempts} attempts: {output.decode("utf-8", errors="replace")}')


def helm(verb, release, chart = None, repo = None, file = None, namespace = None, version = None, wait = False, timeout = None, create_namespace = None, skip_crds = False, atomic = False):
    import subprocess

    cmnd = ['helm', verb, release]
    if not chart is None:
        cmnd.append(chart)
    if verb == 'upgrade':
        cmnd.append('--install')
    if create_namespace:
        cmnd.append('--create-namespace')
    if not repo is None:
        cmnd.extend(['--repo', repo])
    if not file is None:
        cmnd.extend(['--values', file])
    if not version is None:
        cmnd.extend(['--version', version])
    if not namespace is None:
        cmnd.extend(['--namespace', namespace])
    if wait:
        cmnd.append('--wait')
    if skip_crds:
        cmnd.append('--skip-crds')
    if not timeout is None:
        cmnd.extend(['--timeout', timeout])
    if atomic:
        cmnd.append('--atomic')    
    cmnd.extend(['--kubeconfig', kubeconfig])
    
    # Log the full helm command for better troubleshooting
    logger.info("Running command: %s", cmnd)

    maxAttempts = 3
    retry = maxAttempts
    while retry > 0:
        try:
            output = subprocess.check_output(cmnd, stderr=subprocess.STDOUT, cwd=outdir)
            logger.info(output.decode('utf-8', errors='replace'))
            return
        except subprocess.CalledProcessError as exc:
            output = exc.output
            if b'Broken pipe' in output:
                retry = retry - 1
                logger.info("Broken pipe, retries left: %s" % retry)
            else:
                error_message = output.decode('utf-8', errors='replace')
                logger.error("Command failed: %s", cmnd)
                logger.error("Error output: %s", error_message)
                raise Exception(output)
    raise Exception(f'Operation failed after {maxAttempts} attempts: {output.decode("utf-8", errors="replace")}')
