#!/usr/bin/env python

import simplejson as json
import requests
import os

class ClcMiniApi(object):
    """Object to use for making CenturyLink Cloud API calls
    see https://www.ctl.io/api-docs/v2/
    """

    CLC_URL = "https://api.ctl.io/v2/"
    CLC_URL_X = "https://api.ctl.io/v2-experimental/"

    def headers(self):
        headers = {}
        headers['content-type'] = 'application/json; charset=utf8'
        headers['accept'] = 'application/json; charset=utf8'
        if self.bearerToken != None:
            headers['authorization'] = "Bearer " + self.bearerToken
        return headers

    def __init__(self, username, password):
        self.authentication = {}
        self.authentication['username'] = username
        self.authentication['password'] = password
        self.bearerToken = None
        self.accountAlias = None
        self._authenticate()

    def _authenticate(self):
        url = self.CLC_URL + "authentication/login"
        r = requests.post(url, headers=self.headers(),
                          data=json.dumps(self.authentication))
        if r.status_code != 200:
            r.raise_for_status()
        response = json.loads(r.text)
        if response.has_key('accountAlias'):
            self.accountAlias = response["accountAlias"]
        if response.has_key('bearerToken'):
            self.bearerToken = response["bearerToken"]

    def getDatacenter(self, datacenter):
        url = self.CLC_URL + "datacenters/" + self.accountAlias + "/" + datacenter
        params = {'groupLinks': 'true'}
        r = requests.get(url, headers=self.headers(), params=params)
        if r.status_code != 200:
            r.raise_for_status()
        response = json.loads(r.text)
        return response

    def getGroupId(self, groupName, datacenter):
        """In order to get any one name, we must, sadly, pull all of the information
        about every group :( """
        groupIds = {}
        info = self.getDatacenter(datacenter)
        for item in info['links']:
            if item['rel'] == 'group':
                groupId = item['id']
                groupInfo = self.getGroupInfo(groupId)
                groupIds.update(self.getGroupNames(groupInfo))
        if groupIds.has_key(groupName):
            return groupIds[groupName]

    def deleteGroup(self, groupId):
        """Delete the group and all the objects within it."""
        url = self.CLC_URL + "groups/" + self.accountAlias + "/" + groupId
        r = requests.delete(url, headers=self.headers())
        if r.status_code != 200:
            r.raise_for_status()
        response = json.loads(r.text)
        return response

    def getGroupInfo(self, groupId):
        """Get the full groupInfo object for a group, includes detailed subgroup
        and server information """
        url = self.CLC_URL + "groups/" + self.accountAlias + "/" + groupId
        params = {'serverDetail': 'detailed'}
        r = requests.get(url, headers=self.headers(), params=params)
        if r.status_code != 200:
            r.raise_for_status()
        response = json.loads(r.text)
        return response

    def getGroupNames(self, group):
        """ Extract the groupIds as an dict from the nested datastructure """
        group_names_ids = {group["name"]: group["id"]}
        for subgroup in group["groups"]:
            group_names_ids.update(self.getGroupNames(subgroup))
        return group_names_ids

if __name__=="__main__":

    import argparse
    import os
    import shutil

    parser = argparse.ArgumentParser(description='Delete a CLC Kubernetes group and all of its configuration')
    parser.add_argument('--cluster', dest='clc_cluster_name', required=1)
    parser.add_argument('--datacenter', dest='datacenter', required=1)

    args = parser.parse_args()

    try:
        CLC_V2_API_PASSWD = os.environ['CLC_V2_API_PASSWD']
        CLC_V2_API_USERNAME = os.environ['CLC_V2_API_USERNAME']
    except KeyError:
        print "Unable to read CLC username and password from environment variables"
        exit(1)

    local_config = os.environ["HOME"] + "/.clc_kube/" + args.clc_cluster_name

    print "Running this script will permanently delete cluster \""+args.clc_cluster_name+ "\""
    print "and all local configuration files " +local_config 
    print "Continue? [Yn]"
    response=raw_input()

    if response != 'Y':
        exit(0)

    print "now removing " + args.clc_cluster_name

    miniApi = ClcMiniApi(CLC_V2_API_USERNAME, CLC_V2_API_PASSWD)

    groupId = miniApi.getGroupId(args.clc_cluster_name, args.datacenter)

    if groupId != None:
        miniApi.deleteGroup( groupId)
    else:
        print "No group found for "+args.clc_cluster_name+" in datacenter " + args.datacenter

    shutil.rmtree(local_config)
