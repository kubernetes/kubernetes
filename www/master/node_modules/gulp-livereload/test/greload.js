/*jshint strict:false, expr:true, maxlen:999 */
describe('gulp-livereload', function() {
  var gutil = require('gulp-util'),
      sinon = require('sinon'),
      greload = require('..'),
      tinylr = require('tiny-lr'),
      should = require('should');
  beforeEach(function() {
    Object.keys(greload.servers).forEach(function(server) {
      greload.servers[server].close();
    });
    greload.servers = {};
    greload.options = { auto: true };
  });
  it('accepts an lr instance', function(done) {
    var server = tinylr(),
        reload = greload(server),
        mgutil = sinon.mock(gutil),
        mserver = sinon.mock(server);

    mserver.expects('changed').once().withArgs({ body: { files: ['lr.css'] } });
    mgutil.expects('log').once().withArgs(gutil.colors.magenta('lr.css') + ' was reloaded.');

    reload
      .on('data', function() {
        mserver.verify();
        mgutil.verify();
        done();
      })
      .end(new gutil.File({
        path: 'lr.css'
      }));
  });
  it('accepts a port number', function(done) {
    var port = 35730,
        reload = greload(port),
        spy = sinon.spy(),
        mgutil = sinon.mock(gutil),
        mock = sinon.mock(greload);

    mgutil.expects('log').once().withArgs(gutil.colors.magenta('123.css') + ' was reloaded.');
    mock.expects('listen').once().withArgs(port).returns({ changed: spy });

    reload
      .once('data', function() {
        should(spy.calledWith({ body: { files: ['123.css'] } })).ok;
        mock.verify();
        mgutil.verify();
        done();
      })
      .end(new gutil.File({
        path: '123.css'
      }));
  });
  it('requires no parameters', function(done) {
    var reload = greload(),
        spy = sinon.spy(),
        mgutil = sinon.mock(gutil),
        stub = sinon.stub(greload, 'listen');

    mgutil.expects('log').once().withArgs(gutil.colors.magenta('nil.css') + ' was reloaded.');

    stub.withArgs().returns({ changed: spy });
    reload
      .on('data', function() {
        should(spy.calledWith({ body: { files: ['nil.css'] } })).ok;
        mgutil.verify();
        stub.restore();
        done();
      })
      .end(new gutil.File({
        path: 'nil.css'
      }));
  });
  it('doesn\'t display debug messages when in silent mode', function(done) {
    var reload = greload(tinylr(), { silent: true });
    var spy = sinon.spy(gutil, 'log');

    reload
      .on('data', function() {
        spy.called.should.not.be.ok;
        spy.restore();
        done();
      })
      .end(new gutil.File({
        path: 'nil.css'
      }));
  });
  it('exposes tiny-lr middleware', function() {
    (typeof greload.middleware).should.eql('function');
  });
  it('works on https', function(done) {
    require('pem').createCertificate(
      { days: 1, selfSigned: true, keyBitsize: 489 },
      function (err, keys) {
        if (err) done(err);
        var https = require('https');
        var spy = sinon.spy(https, 'createServer');
        greload.listen({
          key: keys.serviceKey,
          cert: keys.certificate,
          silent: true
        });
        spy.calledOnce.should.ok;
        spy.restore();
        done();
      }
    );
  });
  it('won\'t ignite', function(done) {
    greload({ auto: false })
      .once('data', function() {
        should.equal(Object.keys(greload.servers).length, 0);
        should.equal(greload.servers[35730], undefined);
        done();
      })
      .end(new gutil.File({
        path: '123.css'
      }));
  });
  it('manually ignites', function(done) {
    var mgutil = sinon.mock(gutil);

     mgutil.expects('log').once().withArgs('Live reload server listening on: ' + gutil.colors.magenta(35729));
    mgutil.expects('log').once().withArgs(gutil.colors.magenta('ignite.css') + ' was reloaded.');

    greload.listen();
    greload.servers[35729].server.on('listening', function() {
      mgutil.verify();
      done();
    });
    greload({ auto: false }).end(new gutil.File({ path: 'ignite.css' }));
  });
  describe('.changed', function() {
    it('works on strings', function() {
      var spy;

      greload.listen({ silent: true });
      spy = sinon.spy(greload.servers[35729], 'changed');
      greload.changed('str.changed');
      should(spy.calledWith({ body: { files: ['str.changed'] } })).ok;
      spy.restore();
    });
    it('works on objects', function() {
      var spy;

      greload.listen({ silent: true });
      spy = sinon.spy(greload.servers[35729], 'changed');
      greload.changed(new gutil.File({ path: 'obj.changed' }));
      should(spy.calledWith({ body: { files: ['obj.changed'] } })).ok;
      spy.restore();
    });
  });
});
