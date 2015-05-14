#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

import httplib
import json
import time


class Registrator:

    def __init__(self):
        self.ds ={
          "creationTimestamp": "",
          "kind": "Minion",
          "name": "", # private_address
          "metadata": {
            "name": "", #private_address,
          },
          "spec": {
            "externalID": "", #private_address
            "capacity": {
                "mem": "",  # mem + ' K',
                "cpu": "",  # cpus
            }
          },
          "status": {
            "conditions": [],
            "hostIP": "", #private_address
          }
        }

    @property
    def data(self):
        ''' Returns a data-structure for population to make a request. '''
        return self.ds

    def register(self, hostname, port, api_path):
        ''' Contact the API Server for a new registration '''
        headers = {"Content-type": "application/json",
                   "Accept": "application/json"}
        connection = httplib.HTTPConnection(hostname, port)
        print 'CONN {}'.format(connection)
        connection.request("POST", api_path, json.dumps(self.data), headers)
        response = connection.getresponse()
        body = response.read()
        print(body)
        result = json.loads(body)
        print("Response status:%s reason:%s body:%s" % \
             (response.status, response.reason, result))
        return response, result

    def update(self):
        ''' Contact the API Server to update a registration '''
        # do a get on the API for the node
        # repost to the API with any modified data
        pass

    def save(self):
        ''' Marshall the registration data '''
        # TODO
        pass

    def command_succeeded(self, response, result):
        ''' Evaluate response data to determine if the command was successful '''
        if response.status in [200, 201]:
            print("Registered")
            return True
        elif response.status in [409,]:
            print("Status Conflict")
            # Suggested return a PUT instead of a POST with this response
            # code, this predicates use of the UPDATE method
            # TODO
        elif response.status in (500,) and result.get(
            'message', '').startswith('The requested resource does not exist'):
            # There's something fishy in the kube api here (0.4 dev), first time we
            # go to register a new minion, we always seem to get this error.
            # https://github.com/GoogleCloudPlatform/kubernetes/issues/1995
            time.sleep(1)
            print("Retrying registration...")
            raise ValueError("Registration returned 500, retry")
            # return register_machine(apiserver, retry=True)
        else:
            print("Registration error")
            # TODO - get request data
            raise RuntimeError("Unable to register machine with")
