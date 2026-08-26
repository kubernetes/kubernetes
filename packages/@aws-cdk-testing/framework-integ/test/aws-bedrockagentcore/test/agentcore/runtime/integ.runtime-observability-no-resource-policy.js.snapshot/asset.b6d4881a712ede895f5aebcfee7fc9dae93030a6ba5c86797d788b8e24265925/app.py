#!/usr/bin/env python3
"""
Minimal BedrockAgentCore Runtime Test Application
A simple HTTP server that responds to health checks and basic requests
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentRuntimeHandler(BaseHTTPRequestHandler):
    """Simple HTTP request handler for BedrockAgentCore Runtime testing"""
    
    def do_GET(self):
        """Handle GET requests - health check"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'status': 'healthy', 'service': 'bedrock-agentcore-runtime'}
            self.wfile.write(json.dumps(response).encode())
            logger.info("Health check successful")
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests - simulate agent invocation"""
        if self.path == '/invoke':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                # Parse the request
                request_data = json.loads(post_data) if post_data else {}
                prompt = request_data.get('prompt', 'No prompt provided')
                
                # Simple echo response for testing
                response = {
                    'response': f'Echo: {prompt}',
                    'status': 'success',
                    'runtime': 'test-runtime'
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                logger.info(f"Processed request with prompt: {prompt}")
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {'error': str(e), 'status': 'error'}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr"""
        logger.info("%s - %s" % (self.address_string(), format % args))

def run_server(port=8080):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, AgentRuntimeHandler)
    logger.info(f"Starting BedrockAgentCore Runtime test server on port {port}")
    logger.info("Server is ready to handle requests...")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()
