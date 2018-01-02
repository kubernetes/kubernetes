'use strict';

var SwaggerClient = require('swagger-client');

describe('EchoService', function() {
  var client;

  beforeEach(function(done) {
    new SwaggerClient({
      url: "http://localhost:8080/swagger/echo_service.swagger.json",
      usePromise: true,
    }).then(function(c) {
      client = c;
      done();
    });
  });

  describe('Echo', function() {
    it('should echo the request back', function(done) {
      client.EchoService.Echo(
          {id: "foo"},
          {responseContentType: "application/json"}
      ).then(function(resp) {
        expect(resp.obj).toEqual({id: "foo"});
      }).catch(function(err) {
        done.fail(err);
      }).then(done);
    });
  });

  describe('EchoBody', function() {
    it('should echo the request back', function(done) {
      client.EchoService.EchoBody(
          {body: {id: "foo"}},
          {responseContentType: "application/json"}
      ).then(function(resp) {
        expect(resp.obj).toEqual({id: "foo"});
      }).catch(function(err) {
        done.fail(err);
      }).then(done);
    });
  });
});
