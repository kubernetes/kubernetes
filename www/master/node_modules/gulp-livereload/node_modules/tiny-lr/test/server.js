
var request = require('supertest');
var assert  = require('assert');

var Server = require('..').Server;

var listen = require('./helpers/listen');

describe('tiny-lr', function() {

  before(listen());

  describe('GET /', function() {
    it('respond with nothing, but respond', function(done){
      request(this.server)
        .get('/')
        .expect('Content-Type', /json/)
        .expect(/\{"tinylr":"Welcome","version":"[\d].[\d].[\d]+"\}/)
        .expect(200, done);
    });

    it('unknown route respond with proper 404 and error message', function(done){
      request(this.server)
        .get('/whatev')
        .expect('Content-Type', /json/)
        .expect('{"error":"not_found","reason":"no such route"}')
        .expect(404, done);
    });
  });


  describe('GET /changed', function() {
    it('with no clients, no files', function(done) {
      request(this.server)
        .get('/changed')
        .expect('Content-Type', /json/)
        .expect(/"clients":\[\]/)
        .expect(/"files":\[\]/)
        .expect(200, done);
    });

    it('with no clients, some files', function(done) {
      request(this.server)
        .get('/changed?files=gonna.css,test.css,it.css')
        .expect('Content-Type', /json/)
        .expect('{"clients":[],"files":["gonna.css","test.css","it.css"]}')
        .expect(200, done);
    });
  });

  describe('POST /changed', function() {
    it('with no clients, no files', function(done) {
      request(this.server)
        .post('/changed')
        .expect('Content-Type', /json/)
        .expect(/"clients":\[\]/)
        .expect(/"files":\[\]/)
        .expect(200, done);
    });

    it('with no clients, some files', function(done) {
      var data = { clients: [], files: ['cat.css', 'sed.css', 'ack.js'] };

      request(this.server)
        .post('/changed')
        // .type('json')
        .send({ files: data.files })
        .expect('Content-Type', /json/)
        .expect(JSON.stringify(data))
        .expect(200, done);
    });
  });

  describe('GET /livereload.js', function() {
    it('respond with livereload script', function(done) {
      request(this.server)
        .get('/livereload.js')
        .expect(/LiveReload/)
        .expect(200, done);
    });
  });

  describe('GET /kill', function() {
    it('shutdown the server', function(done) {
      var srv = this.server;
      request(srv)
        .get('/kill')
        .expect(200, function(err) {
          if(err) return done(err);
          assert.ok(!srv._handle);
          done();
        });
    });
  });

});
