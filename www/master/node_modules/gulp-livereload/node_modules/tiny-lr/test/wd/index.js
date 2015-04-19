var wd     = require('wd');
var app    = require('../../examples/express/app');
var assert = require('assert');

var port = process.env.LR_PORT || process.env.PORT || 35729;

describe('mocha spec examples', function() {

  this.timeout(10000);

  describe('tinylr', function() {
    var browser;

    before(function(done) {
      var browser = this.browser = wd.remote('localhost', process.env.WD_PORT || 9134);
      browser.init(done)
    });

    before(function(done) {
      this.server = app;
      app.listen(port, done);
    });

    beforeEach(function(done) {
      this.browser.get('http://localhost:' + port, done);
    });

    it('should retrieve the page title', function(done) {
      this.browser.title(function(err, title) {
        if (err) return done(err);
        assert.equal(title, 'WD Tests');
        done();
      });
    });

    it('edit file, assert change');
  });

});
