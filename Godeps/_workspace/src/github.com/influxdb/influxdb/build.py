#!/usr/bin/env python2.7
#
# This is the InfluxDB build script.
#
# Current caveats:
#   - Does not currently build ARM builds/packages
#   - Does not checkout the correct commit/branch (for now, you will need to do so manually)
#   - Has external dependencies for packaging (fpm) and uploading (boto)
#

import sys
import os
import subprocess
import time
import datetime
import shutil
import tempfile
import hashlib
import re

try:
    import boto
    from boto.s3.key import Key
except ImportError:
    pass

# PACKAGING VARIABLES
INSTALL_ROOT_DIR = "/usr/bin"
LOG_DIR = "/var/log/influxdb"
DATA_DIR = "/var/lib/influxdb"
SCRIPT_DIR = "/usr/lib/influxdb/scripts"
CONFIG_DIR = "/etc/influxdb"
LOGROTATE_DIR = "/etc/logrotate.d"

INIT_SCRIPT = "scripts/init.sh"
SYSTEMD_SCRIPT = "scripts/influxdb.service"
PREINST_SCRIPT = "scripts/pre-install.sh"
POSTINST_SCRIPT = "scripts/post-install.sh"
POSTUNINST_SCRIPT = "scripts/post-uninstall.sh"
LOGROTATE_SCRIPT = "scripts/logrotate"
DEFAULT_CONFIG = "etc/config.sample.toml"

# META-PACKAGE VARIABLES
PACKAGE_LICENSE = "MIT"
PACKAGE_URL = "https://github.com/influxdb/influxdb"
MAINTAINER = "InfluxData"
VENDOR = "InfluxData"
DESCRIPTION = "A distributed time-series database."

# SCRIPT START
prereqs = [ 'git', 'go' ]
optional_prereqs = [ 'gvm', 'fpm', 'rpmbuild' ]

fpm_common_args = "-f -s dir --log error \
 --vendor {} \
 --url {} \
 --after-install {} \
 --before-install {} \
 --after-remove {} \
 --license {} \
 --maintainer {} \
 --config-files {} \
 --config-files {} \
 --description \"{}\"".format(
    VENDOR,
    PACKAGE_URL,
    POSTINST_SCRIPT,
    PREINST_SCRIPT,
    POSTUNINST_SCRIPT,
    PACKAGE_LICENSE,
    MAINTAINER,
    CONFIG_DIR,
    LOGROTATE_DIR,
    DESCRIPTION)

targets = {
    'influx' : './cmd/influx/main.go',
    'influxd' : './cmd/influxd/main.go',
    'influx_stress' : './cmd/influx_stress/influx_stress.go',
    'influx_inspect' : './cmd/influx_inspect/*.go',
}

supported_builds = {
    # TODO(rossmcdonald): Add support for multiple GOARM values
    'darwin': [ "amd64", "386" ],
    # Windows is not currently supported in InfluxDB 0.9.5 due to use of mmap
    # 'windows': [ "amd64", "386", "arm" ],
    'linux': [ "amd64", "386", "arm" ]
}
supported_go = [ '1.5.1', '1.4.2' ]
supported_packages = {
    "darwin": [ "tar", "zip" ],
    "linux": [ "deb", "rpm", "tar", "zip" ],
    "windows": [ "tar", "zip" ],
}

def run(command, allow_failure=False, shell=False):
    out = None
    try:
        if shell:
            out = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=shell)
        else:
            out = subprocess.check_output(command.split(), stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print ""
        print "Executed command failed!"
        print "-- Command run was: {}".format(command)
        print "-- Failure was: {}".format(e.output)
        if allow_failure:
            print "Continuing..."
            return out
        else:
            print ""
            print "Stopping."
            sys.exit(1)
    except OSError as e:
        print ""
        print "Invalid command!"
        print "-- Command run was: {}".format(command)
        print "-- Failure was: {}".format(e)
        if allow_failure:
            print "Continuing..."
            return out
        else:
            print ""
            print "Stopping."
            sys.exit(1)
    else:
        return out

def create_temp_dir():
    return tempfile.mkdtemp(prefix="influxdb-build.")

def get_current_version():
    command = "git describe --always --tags --abbrev=0"
    out = run(command)
    return out.strip()

def get_current_commit(short=False):
    command = None
    if short:
        command = "git log --pretty=format:'%h' -n 1"
    else:
        command = "git rev-parse HEAD"
    out = run(command)
    return out.strip('\'\n\r ')

def get_current_branch():
    command = "git rev-parse --abbrev-ref HEAD"
    out = run(command)
    return out.strip()

def get_system_arch():
    return os.uname()[4]

def get_system_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    else:
        return sys.platform

def get_go_version():
    out = run("go version")
    matches = re.search('go version go(\S+)', out)
    if matches is not None:
        return matches.groups()[0].strip()
    return None

def check_path_for(b):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        full_path = os.path.join(path, b)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            return full_path

def check_environ(build_dir = None):
    print "\nChecking environment:"
    for v in [ "GOPATH", "GOBIN" ]:
        print "\t- {} -> {}".format(v, os.environ.get(v))

    cwd = os.getcwd()
    if build_dir == None and os.environ.get("GOPATH") and os.environ.get("GOPATH") not in cwd:
        print "\n!! WARNING: Your current directory is not under your GOPATH! This probably won't work."

def check_prereqs():
    print "\nChecking for dependencies:"
    for req in prereqs:
        print "\t- {} ->".format(req),
        path = check_path_for(req)
        if path:
            print "{}".format(path)
        else:
            print "?"
    for req in optional_prereqs:
        print "\t- {} (optional) ->".format(req),
        path = check_path_for(req)
        if path:
            print "{}".format(path)
        else:
            print "?"
    print ""

def upload_packages(packages, nightly=False):
    print "Uploading packages to S3..."
    print ""
    c = boto.connect_s3()
    # TODO(rossmcdonald) - Set to different S3 bucket for release vs nightly
    bucket = c.get_bucket('influxdb-nightly')
    for p in packages:
        name = os.path.basename(p)
        if bucket.get_key(name) is None or nightly:
            print "\t - Uploading {}...".format(name),
            k = Key(bucket)
            k.key = name
            if nightly:
                n = k.set_contents_from_filename(p, replace=True)
            else:
                n = k.set_contents_from_filename(p, replace=False)
            k.make_public()
            print "[ DONE ]"
        else:
            print "\t - Not uploading {}, already exists.".format(p)
    print ""

def run_tests(race, parallel, timeout, no_vet):
    get_command = "go get -d -t ./..."
    print "Retrieving Go dependencies...",
    sys.stdout.flush()
    run(get_command)
    print "done."
    print "Running tests:"
    print "\tRace: ", race
    if parallel is not None:
        print "\tParallel:", parallel
    if timeout is not None:
        print "\tTimeout:", timeout
    sys.stdout.flush()
    p = subprocess.Popen(["go", "fmt", "./..."], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if len(out) > 0 or len(err) > 0:
        print "Code not formatted. Please use 'go fmt ./...' to fix formatting errors."
        print out
        print err
        return False
    if not no_vet:
        p = subprocess.Popen(["go", "tool", "vet", "-composites=false", "./"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if len(out) > 0 or len(err) > 0:
            print "Go vet failed. Please run 'go vet ./...' and fix any errors."
            print out
            print err
            return False
    else:
        print "Skipping go vet ..."
        sys.stdout.flush()
    test_command = "go test -v"
    if race:
        test_command += " -race"
    if parallel is not None:
        test_command += " -parallel {}".format(parallel)
    if timeout is not None:
        test_command += " -timeout {}".format(timeout)
    test_command += " ./..."
    code = os.system(test_command)
    if code != 0:
        print "Tests Failed"
        return False
    else:
        print "Tests Passed"
        return True

def build(version=None,
          branch=None,
          commit=None,
          platform=None,
          arch=None,
          nightly=False,
          rc=None,
          race=False,
          clean=False,
          outdir="."):
    print "-------------------------"
    print ""
    print "Build plan:"
    print "\t- version: {}".format(version)
    if rc:
        print "\t- release candidate: {}".format(rc)
    print "\t- commit: {}".format(commit)
    print "\t- branch: {}".format(branch)
    print "\t- platform: {}".format(platform)
    print "\t- arch: {}".format(arch)
    print "\t- nightly? {}".format(str(nightly).lower())
    print "\t- race enabled? {}".format(str(race).lower())
    print ""

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif clean and outdir != '/':
        print "Cleaning build directory..."
        shutil.rmtree(outdir)
        os.makedirs(outdir)

    if rc:
        # If a release candidate, update the version information accordingly
        version = "{}rc{}".format(version, rc)

    print "Starting build..."
    for b, c in targets.iteritems():
        print "\t- Building '{}'...".format(os.path.join(outdir, b)),
        build_command = ""
        build_command += "GOOS={} GOOARCH={} ".format(platform, arch)
        if arch == "arm":
            # TODO(rossmcdonald): Add GOARM variables for ARM builds
            build_command += "GOOARM={} ".format(6)
        build_command += "go build -o {} ".format(os.path.join(outdir, b))
        if race:
            build_command += "-race "
        go_version = get_go_version()
        if "1.4" in go_version:
            build_command += "-ldflags=\"-X main.buildTime '{}' ".format(datetime.datetime.utcnow().isoformat())
            build_command += "-X main.version {} ".format(version)
            build_command += "-X main.branch {} ".format(branch)
            build_command += "-X main.commit {}\" ".format(get_current_commit())
        else:
            build_command += "-ldflags=\"-X main.buildTime='{}' ".format(datetime.datetime.utcnow().isoformat())
            build_command += "-X main.version={} ".format(version)
            build_command += "-X main.branch={} ".format(branch)
            build_command += "-X main.commit={}\" ".format(get_current_commit())
        build_command += c
        out = run(build_command, shell=True)
        print "[ DONE ]"
    print ""

def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print e

def rename_file(fr, to):
    try:
        os.rename(fr, to)
    except OSError as e:
        print e
        # Return the original filename
        return fr
    else:
        # Return the new filename
        return to

def copy_file(fr, to):
    try:
        shutil.copy(fr, to)
    except OSError as e:
        print e

def create_package_fs(build_root):
    print "\t- Creating a filesystem hierarchy from directory: {}".format(build_root)
    # Using [1:] for the path names due to them being absolute
    # (will overwrite previous paths, per 'os.path.join' documentation)
    dirs = [ INSTALL_ROOT_DIR[1:], LOG_DIR[1:], DATA_DIR[1:], SCRIPT_DIR[1:], CONFIG_DIR[1:], LOGROTATE_DIR[1:] ]
    for d in dirs:
        create_dir(os.path.join(build_root, d))
        os.chmod(os.path.join(build_root, d), 0755)

def package_scripts(build_root):
    print "\t- Copying scripts and sample configuration to build directory"
    shutil.copyfile(INIT_SCRIPT, os.path.join(build_root, SCRIPT_DIR[1:], INIT_SCRIPT.split('/')[1]))
    os.chmod(os.path.join(build_root, SCRIPT_DIR[1:], INIT_SCRIPT.split('/')[1]), 0644)
    shutil.copyfile(SYSTEMD_SCRIPT, os.path.join(build_root, SCRIPT_DIR[1:], SYSTEMD_SCRIPT.split('/')[1]))
    os.chmod(os.path.join(build_root, SCRIPT_DIR[1:], SYSTEMD_SCRIPT.split('/')[1]), 0644)
    shutil.copyfile(LOGROTATE_SCRIPT, os.path.join(build_root, SCRIPT_DIR[1:], LOGROTATE_SCRIPT.split('/')[1]))
    os.chmod(os.path.join(build_root, SCRIPT_DIR[1:], LOGROTATE_SCRIPT.split('/')[1]), 0644)
    shutil.copyfile(DEFAULT_CONFIG, os.path.join(build_root, CONFIG_DIR[1:], "influxdb.conf"))
    os.chmod(os.path.join(build_root, CONFIG_DIR[1:], "influxdb.conf"), 0644)

def go_get(update=False):
    get_command = None
    if update:
        get_command = "go get -u -f -d ./..."
    else:
        get_command = "go get -d ./..."
    print "Retrieving Go dependencies...",
    run(get_command)
    print "done.\n"

def generate_md5_from_file(path):
    m = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()

def build_packages(build_output, version, nightly=False, rc=None, iteration=1):
    outfiles = []
    tmp_build_dir = create_temp_dir()
    try:
        print "-------------------------"
        print ""
        print "Packaging..."
        for p in build_output:
            # Create top-level folder displaying which platform (linux, etc)
            create_dir(os.path.join(tmp_build_dir, p))
            for a in build_output[p]:
                current_location = build_output[p][a]
                # Create second-level directory displaying the architecture (amd64, etc)p
                build_root = os.path.join(tmp_build_dir, p, a)
                # Create directory tree to mimic file system of package
                create_dir(build_root)
                create_package_fs(build_root)
                # Copy in packaging and miscellaneous scripts
                package_scripts(build_root)
                # Copy newly-built binaries to packaging directory
                for b in targets:
                    if p == 'windows':
                        b = b + '.exe'
                    fr = os.path.join(current_location, b)
                    to = os.path.join(build_root, INSTALL_ROOT_DIR[1:], b)
                    print "\t- [{}][{}] - Moving from '{}' to '{}'".format(p, a, fr, to)
                    copy_file(fr, to)
                # Package the directory structure
                for package_type in supported_packages[p]:
                    print "\t- Packaging directory '{}' as '{}'...".format(build_root, package_type),
                    name = "influxdb"
                    if package_type in ['zip', 'tar']:
                        if nightly:
                            name = '{}-nightly_{}_{}'.format(name, p, a)
                        else:
                            name = '{}-{}_{}_{}'.format(name, version, p, a)
                    if package_type == 'tar':
                        # Add `tar.gz` to path to ensure a small package size
                        current_location = os.path.join(current_location, name + '.tar.gz')
                    if package_type == 'deb' and rc:
                        # For debs with an RC, just append to version number
                        version += "-rc{}".format(rc)
                    fpm_command = "fpm {} --name {} -a {} -t {} --version {} -C {} -p {} ".format(
                        fpm_common_args,
                        name,
                        a,
                        package_type,
                        version,
                        build_root,
                        current_location)
                    if package_type == "rpm":
                        fpm_command += "--depends coreutils "
                        # For rpms with RC, add to iteration for adherence to Fedora packaging standard:
                        # http://fedoraproject.org/wiki/Packaging%3aNamingGuidelines#NonNumericRelease
                        if rc:
                            fpm_command += "--iteration 0.{}.rc{} ".format(iteration, rc)
                        else:
                            fpm_command += "--iteration 1 ".format(iteration)
                    out = run(fpm_command, shell=True)
                    matches = re.search(':path=>"(.*)"', out)
                    outfile = None
                    if matches is not None:
                        outfile = matches.groups()[0]
                    if outfile is None:
                        print "[ COULD NOT DETERMINE OUTPUT ]"
                    else:
                        # Strip nightly version (the unix epoch) from filename
                        if nightly and package_type == 'deb':
                            outfile = rename_file(outfile, outfile.replace("{}".format(version), "nightly"))
                        elif nightly and package_type == 'rpm':
                            outfile = rename_file(outfile, outfile.replace("{}-1".format(version), "nightly"))
                        outfiles.append(os.path.join(os.getcwd(), outfile))
                        print "[ DONE ]"
                        # Display MD5 hash for generated package
                        print "\t\tMD5 = {}".format(generate_md5_from_file(outfile))
        print ""
        return outfiles
    finally:
        # Cleanup
        shutil.rmtree(tmp_build_dir)

def print_usage():
    print "Usage: ./build.py [options]"
    print ""
    print "Options:"
    print "\t --outdir=<path> \n\t\t- Send build output to a specified path. Defaults to ./build."
    print "\t --arch=<arch> \n\t\t- Build for specified architecture. Acceptable values: x86_64|amd64, 386, arm, or all"
    print "\t --platform=<platform> \n\t\t- Build for specified platform. Acceptable values: linux, windows, darwin, or all"
    print "\t --version=<version> \n\t\t- Version information to apply to build metadata. If not specified, will be pulled from repo tag."
    print "\t --commit=<commit> \n\t\t- Use specific commit for build (currently a NOOP)."
    print "\t --branch=<branch> \n\t\t- Build from a specific branch (currently a NOOP)."
    print "\t --rc=<rc number> \n\t\t- Whether or not the build is a release candidate (affects version information)."
    print "\t --race \n\t\t- Whether the produced build should have race detection enabled."
    print "\t --package \n\t\t- Whether the produced builds should be packaged for the target platform(s)."
    print "\t --nightly \n\t\t- Whether the produced build is a nightly (affects version information)."
    print "\t --update \n\t\t- Whether dependencies should be updated prior to building."
    print "\t --test \n\t\t- Run Go tests. Will not produce a build."
    print "\t --parallel \n\t\t- Run Go tests in parallel up to the count specified."
    print "\t --timeout \n\t\t- Timeout for Go tests. Default 480s"
    print "\t --clean \n\t\t- Clean the build output directory prior to creating build."
    print ""

def print_package_summary(packages):
    print packages

def main():
    # Command-line arguments
    outdir = "build"
    commit = None
    target_platform = None
    target_arch = None
    nightly = False
    race = False
    branch = None
    version = get_current_version()
    rc = None
    package = False
    update = False
    clean = False
    upload = False
    test = False
    parallel = None
    timeout = None
    iteration = 1
    no_vet = False

    for arg in sys.argv[1:]:
        if '--outdir' in arg:
            # Output directory. If none is specified, then builds will be placed in the same directory.
            output_dir = arg.split("=")[1]
        if '--commit' in arg:
            # Commit to build from. If none is specified, then it will build from the most recent commit.
            commit = arg.split("=")[1]
        if '--branch' in arg:
            # Branch to build from. If none is specified, then it will build from the current branch.
            branch = arg.split("=")[1]
        elif '--arch' in arg:
            # Target architecture. If none is specified, then it will build for the current arch.
            target_arch = arg.split("=")[1]
        elif '--platform' in arg:
            # Target platform. If none is specified, then it will build for the current platform.
            target_platform = arg.split("=")[1]
        elif '--version' in arg:
            # Version to assign to this build (0.9.5, etc)
            version = arg.split("=")[1]
        elif '--rc' in arg:
            # Signifies that this is a release candidate build.
            rc = arg.split("=")[1]
        elif '--race' in arg:
            # Signifies that race detection should be enabled.
            race = True
        elif '--package' in arg:
            # Signifies that packages should be built.
            package = True
        elif '--nightly' in arg:
            # Signifies that this is a nightly build.
            nightly = True
        elif '--update' in arg:
            # Signifies that dependencies should be updated.
            update = True
        elif '--upload' in arg:
            # Signifies that the resulting packages should be uploaded to S3
            upload = True
        elif '--test' in arg:
            # Run tests and exit
            test = True
        elif '--parallel' in arg:
            # Set parallel for tests.
            parallel = int(arg.split("=")[1])
        elif '--timeout' in arg:
            # Set timeout for tests.
            timeout = arg.split("=")[1]
        elif '--clean' in arg:
            # Signifies that the outdir should be deleted before building
            clean = True
        elif '--iteration' in arg:
            iteration = arg.split("=")[1]
        elif '--no-vet' in arg:
            no_vet = True
        elif '--help' in arg:
            print_usage()
            return 0
        else:
            print "!! Unknown argument: {}".format(arg)
            print_usage()
            return 1

    if nightly:
        if rc:
            print "!! Cannot be both nightly and a release candidate! Stopping."
            return 1
        # In order to support nightly builds on the repository, we are adding the epoch timestamp
        # to the version so that version numbers are always greater than the previous nightly.
        version = "{}.n{}".format(version, int(time.time()))

    # Pre-build checks
    check_environ()
    check_prereqs()

    if not commit:
        commit = get_current_commit(short=True)
    if not branch:
        branch = get_current_branch()
    if not target_arch:
        target_arch = get_system_arch()
    if not target_platform:
        target_platform = get_system_platform()

    if target_arch == "x86_64":
        target_arch = "amd64"

    build_output = {}
    # TODO(rossmcdonald): Prepare git repo for build (checking out correct branch/commit, etc.)
    # prepare(branch=branch, commit=commit)
    if test:
        if not run_tests(race, parallel, timeout, no_vet):
            return 1
        return 0

    go_get(update=update)

    platforms = []
    single_build = True
    if target_platform == 'all':
        platforms = supported_builds.keys()
        single_build = False
    else:
        platforms = [target_platform]

    for platform in platforms:
        build_output.update( { platform : {} } )
        archs = []
        if target_arch == "all":
            single_build = False
            archs = supported_builds.get(platform)
        else:
            archs = [target_arch]
        for arch in archs:
            od = outdir
            if not single_build:
                od = os.path.join(outdir, platform, arch)
            build(version=version,
                  branch=branch,
                  commit=commit,
                  platform=platform,
                  arch=arch,
                  nightly=nightly,
                  rc=rc,
                  race=race,
                  clean=clean,
                  outdir=od)
            build_output.get(platform).update( { arch : od } )

    # Build packages
    if package:
        if not check_path_for("fpm"):
            print "!! Cannot package without command 'fpm'. Stopping."
            return 1
        packages = build_packages(build_output, version, nightly=nightly, rc=rc, iteration=iteration)
        # TODO(rossmcdonald): Add nice output for print_package_summary()
        # print_package_summary(packages)
        # Optionally upload to S3
        if upload:
            upload_packages(packages, nightly=nightly)
    return 0

if __name__ == '__main__':
    sys.exit(main())

