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

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

PORT_NUMBER = 8000

# This class will handles any incoming request.
class HTTPHandler(BaseHTTPRequestHandler):
  # Handler for the GET requests
  def do_GET(self):
    self.send_response(200)
    self.send_header('Content-type','text/html')
    self.end_headers()
    self.wfile.write("Hello World!")

try:
  # Create a web server and define the handler to manage the incoming request.
  server = HTTPServer(('', PORT_NUMBER), HTTPHandler)
  print 'Started httpserver on port ' , PORT_NUMBER
  server.serve_forever()
except KeyboardInterrupt:
  print '^C received, shutting down the web server'
  server.socket.close()
