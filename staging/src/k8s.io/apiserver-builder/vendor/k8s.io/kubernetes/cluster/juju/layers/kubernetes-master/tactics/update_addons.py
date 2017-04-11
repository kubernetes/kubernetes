#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import subprocess
import tempfile
import logging
from contextlib import contextmanager

import charmtools.utils
from charmtools.build.tactics import Tactic


description = """
Update addon manifests for the charm.

This will clone the kubernetes repo and place the addons in
<charm>/templates/addons.

Can be run with no arguments and from any folder.
"""

log = logging.getLogger(__name__)


def clean_addon_dir(addon_dir):
    """ Remove and recreate the addons folder """
    log.debug("Cleaning " + addon_dir)
    shutil.rmtree(addon_dir, ignore_errors=True)
    os.makedirs(addon_dir)


def run_with_logging(command):
    """ Run a command with controlled logging """
    log.debug("Running: %s" % command)
    process = subprocess.Popen(command, stderr=subprocess.PIPE)
    stderr = process.communicate()[1].rstrip()
    process.wait()
    if process.returncode != 0:
        log.error(stderr)
        raise Exception("%s: exit code %d" % (command, process.returncode))
    log.debug(stderr)


@contextmanager
def kubernetes_repo():
    """ Yield a kubernetes repo to copy addons from.

    If KUBE_VERSION is set, this will clone the local repo and checkout the
    corresponding branch. Otherwise, the local branch will be used. """
    repo = os.path.abspath("../../../..")
    if "KUBE_VERSION" in os.environ:
        branch = os.environ["KUBE_VERSION"]
        log.info("Cloning %s with branch %s" % (repo, branch))
        path = tempfile.mkdtemp(prefix="kubernetes")
        try:
            cmd = ["git", "clone", repo, path, "-b", branch]
            run_with_logging(cmd)
            yield path
        finally:
            shutil.rmtree(path)
    else:
        log.info("Using local repo " + repo)
        yield repo


def add_addon(repo, source, dest):
    """ Add an addon manifest from the given repo and source.

    Any occurrences of 'amd64' are replaced with '{{ arch }}' so the charm can
    fill it in during deployment. """
    source = os.path.join(repo, "cluster/addons", source)
    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(source))
    log.debug("Copying: %s -> %s" % (source, dest))
    with open(source, "r") as f:
        content = f.read()
    content = content.replace("amd64", "{{ arch }}")
    with open(dest, "w") as f:
        f.write(content)


def update_addons(dest):
    """ Update addons. This will clean the addons folder and add new manifests
    from upstream. """
    with kubernetes_repo() as repo:
        log.info("Copying addons to charm")
        clean_addon_dir(dest)
        add_addon(repo, "dashboard/dashboard-controller.yaml", dest)
        add_addon(repo, "dashboard/dashboard-service.yaml", dest)
        try:
            add_addon(repo, "dns/kubedns-sa.yaml",
                      dest + "/kubedns-sa.yaml")
            add_addon(repo, "dns/kubedns-cm.yaml",
                      dest + "/kubedns-cm.yaml")
            add_addon(repo, "dns/kubedns-controller.yaml.in",
                      dest + "/kubedns-controller.yaml")
            add_addon(repo, "dns/kubedns-svc.yaml.in",
                      dest + "/kubedns-svc.yaml")
        except IOError as e:
            # fall back to the older filenames
            log.debug(e)
            add_addon(repo, "dns/skydns-rc.yaml.in",
                      dest + "/kubedns-controller.yaml")
            add_addon(repo, "dns/skydns-svc.yaml.in",
                      dest + "/kubedns-svc.yaml")
        influxdb = "cluster-monitoring/influxdb"
        add_addon(repo, influxdb + "/grafana-service.yaml", dest)
        add_addon(repo, influxdb + "/heapster-controller.yaml", dest)
        add_addon(repo, influxdb + "/heapster-service.yaml", dest)
        add_addon(repo, influxdb + "/influxdb-grafana-controller.yaml", dest)
        add_addon(repo, influxdb + "/influxdb-service.yaml", dest)

# Entry points


class UpdateAddonsTactic(Tactic):
    """ This tactic is used by charm-tools to dynamically populate the
    template/addons folder at `charm build` time. """

    @classmethod
    def trigger(cls, entity, target=None, layer=None, next_config=None):
        """ Determines which files the tactic should apply to. We only want
        this tactic to trigger once, so let's use the templates/ folder
        """
        relpath = entity.relpath(layer.directory) if layer else entity
        return relpath == "templates"

    @property
    def dest(self):
        """ The destination we are writing to. This isn't a Tactic thing,
        it's just a helper for UpdateAddonsTactic """
        return self.target / "templates" / "addons"

    def __call__(self):
        """ When the tactic is called, update addons and put them directly in
        our build destination """
        update_addons(self.dest)

    def sign(self):
        """ Return signatures for the charm build manifest. We need to do this
        because the addon template files were added dynamically """
        sigs = {}
        for file in os.listdir(self.dest):
            path = self.dest / file
            relpath = path.relpath(self.target.directory)
            sigs[relpath] = (
                self.current.url,
                "dynamic",
                charmtools.utils.sign(path)
            )
        return sigs


def parse_args():
    """ Parse args. This is solely done for the usage output with -h """
    parser = argparse.ArgumentParser(description=description)
    parser.parse_args()


def main():
    """ Update addons into the layer's templates/addons folder """
    parse_args()
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    dest = "templates/addons"
    update_addons(dest)


if __name__ == "__main__":
    main()
