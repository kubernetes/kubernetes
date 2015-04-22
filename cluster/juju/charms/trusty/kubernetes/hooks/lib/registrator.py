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
