describe('SauceLabs', function () {
  var SauceLabs = require('..');
  var sauce;
  var nockle;

  describe('#constructor', function () {
    it('can be instantiated with `new`', function () {
      sauce = new SauceLabs();
      sauce.should.be.an.instanceof(SauceLabs);
    });
  });

  describe('once instantiated', function () {
    beforeEach(function () {
      var base = 'https://:username::password@saucelabs.com';
      var config = {
        username: 'johndoe',
        password: '550e8400-e29b-41d4-a716-446655440000'
      };

      sauce  = new SauceLabs(config);
      nockle = new chai.Nockle(base, config);
    });

    afterEach(function () {
      sauce  = null;
      nockle = null;
    });

    describe('when the response status is not 200', function () {
      it('notifies the callback', function (done) {
        var error = { error: 'foobar' };
        var mock = nockle.failget('/rest/v1/users/:username', null, error);
        sauce.getAccountDetails(verifyFailure(error, done));
      });
    });

    describe('when the response body cannot be parsed', function () {
      it('notifies the callback', function (done) {
        var error = 'foobar';
        var mock = nockle.failget('/rest/v1/users/:username', null, error);
        sauce.getAccountDetails(verifyFailure('Could not parse response: ' + error, done));
      });
    });

    describe('#getAccountDetails', function () {
      it('GETs `/rest/v1/users/:username`', function (done) {
        var mock = nockle.get('/rest/v1/users/:username');
        sauce.getAccountDetails(verifySuccess(mock, done));
      });
    });

    describe('#getAccountLimits', function () {
      it('GETs `/rest/v1/:username/limits`', function (done) {
        var mock = nockle.get('/rest/v1/:username/limits');
        sauce.getAccountLimits(verifySuccess(mock, done));
      });
    });

    describe('#getUserActivity', function () {
      describe('without start and end dates', function () {
        it('GETs `/rest/v1/:username/activity`', function (done) {
          var mock = nockle.get('/rest/v1/:username/activity');
          sauce.getUserActivity(verifySuccess(mock, done));
        });
      });

      describe('with start date', function () {
        var start = new Date('Jan 1, 1970 00:00:00');

        it('GETs `/rest/v1/:username/activity?start=1970-01-01`', function (done) {
          var mock = nockle.get('/rest/v1/:username/activity?start=1970-01-01');
          sauce.getUserActivity(start, verifySuccess(mock, done));
        });
      });

      describe('with end date', function () {
        var end = new Date('Jan 1, 1971 00:00:00');

        it('GETs `/rest/v1/:username/activity?end=1971-01-01`', function (done) {
          var mock = nockle.get('/rest/v1/:username/activity?end=1971-01-01');
          sauce.getUserActivity(null, end, verifySuccess(mock, done));
        });
      });

      describe('with start and end dates', function () {
        var start = new Date('Jan 1, 1970 00:00:00');
        var end   = new Date('Jan 1, 1971 00:00:00');

        it('GETs `/rest/v1/:username/activity?start=1970-01-01&end=1971-01-01`', function (done) {
          var mock = nockle.get('/rest/v1/:username/activity?start=1970-01-01&end=1971-01-01');
          sauce.getUserActivity(start, end, verifySuccess(mock, done));
        });
      });
    });

    describe('#getAccountUsage', function () {
      it('GETs `/rest/v1/users/:username/usage`', function (done) {
        var mock = nockle.get('/rest/v1/users/:username/usage');
        sauce.getAccountUsage(verifySuccess(mock, done));
      });
    });

    describe('#getJobs', function () {
      it('GETs `/rest/v1/:username/jobs?full=true`', function (done) {
        var mock = nockle.get('/rest/v1/:username/jobs?full=true');
        sauce.getJobs(verifySuccess(mock, done));
      });
    });

    describe('#showJob', function () {
      it('GETs `/rest/v1/:username/jobs/:id`', function (done) {
        var id = '01230123-example-id-1234';
        var mock = nockle.get('/rest/v1/:username/jobs/:id', { id: id });
        sauce.showJob(id, verifySuccess(mock, done));
      });
    });

    describe('#updateJob', function () {
      it('PUTs `/rest/v1/:username/jobs/:id`', function (done) {
        var id = '01230123-example-id-1234';
        var mock = nockle.put('/rest/v1/:username/jobs/:id', { id: id });
        sauce.updateJob(id, {}, verifySuccess(mock, done));
      });
    });

    describe('#stopJob', function () {
      it('PUTs `/rest/v1/:username/jobs/:id/stop`', function (done) {
        var id = '01230123-example-id-1234';
        var mock = nockle.put('/rest/v1/:username/jobs/:id/stop', { id: id });
        sauce.stopJob(id, {}, verifySuccess(mock, done));
      });
    });

    describe('#getActiveTunnels', function () {
      it('GETs `/rest/v1/:username/tunnels`', function (done) {
        var mock = nockle.get('/rest/v1/:username/tunnels');
        sauce.getActiveTunnels(verifySuccess(mock, done));
      });
    });

    describe('#getTunnel', function () {
      it('GETs `/rest/v1/:username/tunnels/:id`', function (done) {
        var id = '01230123-example-id-1234';
        var mock = nockle.get('/rest/v1/:username/tunnels/:id', { id: id });
        sauce.getTunnel(id, verifySuccess(mock, done));
      });
    });

    describe('#deleteTunnel', function () {
      it('DELETEs `/rest/v1/:username/tunnels/:id`', function (done) {
        var id = '01230123-example-id-1234';
        var mock = nockle.delete('/rest/v1/:username/tunnels/:id', { id: id });
        sauce.deleteTunnel(id, verifySuccess(mock, done));
      });
    });

    describe('#getServiceStatus', function () {
      it('GETs `/rest/v1/info/status`', function (done) {
        var mock = nockle.get('/rest/v1/info/status');
        sauce.getServiceStatus(verifySuccess(mock, done));
      });
    });

    describe('#getBrowsers', function () {
      it('GETs `/rest/v1/info/browsers`', function (done) {
        var mock = nockle.get('/rest/v1/info/browsers');
        sauce.getBrowsers(verifySuccess(mock, done));
      });
    });

    describe('#getAllBrowsers', function () {
      it('GETs `/rest/v1/info/browsers/all`', function (done) {
        var mock = nockle.get('/rest/v1/info/browsers/all');
        sauce.getAllBrowsers(verifySuccess(mock, done));
      });
    });

    describe('#getSeleniumBrowsers', function () {
      it('GETs `/rest/v1/info/browsers/selenium-rc`', function (done) {
        var mock = nockle.get('/rest/v1/info/browsers/selenium-rc');
        sauce.getSeleniumBrowsers(verifySuccess(mock, done));
      });
    });

    describe('#getWebDriverBrowsers', function () {
      it('GETs `/rest/v1/info/browsers/webdriver`', function (done) {
        var mock = nockle.get('/rest/v1/info/browsers/webdriver');
        sauce.getWebDriverBrowsers(verifySuccess(mock, done));
      });
    });

    describe('#getTestCounter', function () {
      it('GETs `/rest/v1/info/counter`', function (done) {
        var mock = nockle.get('/rest/v1/info/counter');
        sauce.getTestCounter(verifySuccess(mock, done));
      });
    });

    describe('#createSubAccount', function () {
      it('POSTs `/rest/v1/users/:username`', function (done) {
        var mock = nockle.post('/rest/v1/users/:username');
        sauce.createSubAccount({}, verifySuccess(mock, done));
      });
    });

    describe('#updateSubAccount', function () {
      it('POSTs `/rest/v1/users/:username/subscription`', function (done) {
        var mock = nockle.post('/rest/v1/users/:username/subscription');
        sauce.updateSubAccount({}, verifySuccess(mock, done));
      });
    });

    describe('#deleteSubAccount', function () {
      it('DELETEs `/rest/v1/users/:username/subscription`', function (done) {
        var mock = nockle.delete('/rest/v1/users/:username/subscription');
        sauce.deleteSubAccount(verifySuccess(mock, done));
      });
    });

    describe('#createPublicLink', function () {
      var id = '01230123-example-id-1234';
      var date = new Date('Jan 1, 1970 00:00:00');

      describe('with job ID, date and hour', function () {
        it('creates the proper link', function (done) {
          var expected = 'https://saucelabs.com/jobs/01230123-example-id-1234?auth=8bcfe0b2e888794a63050ea74ad12005';
          sauce.createPublicLink(id, date, true, verifyLink(expected, done));
        });
      });

      describe('with job ID and date', function () {
        it('creates the proper link', function (done) {
          var expected = 'https://saucelabs.com/jobs/01230123-example-id-1234?auth=c86f247871504dfc1b11736b334b2409';
          sauce.createPublicLink(id, date, verifyLink(expected, done));
        });
      });

      describe('with job ID', function () {
        it('creates the proper link', function (done) {
          var expected = 'https://saucelabs.com/jobs/01230123-example-id-1234?auth=211e7791e357ddbc22065b864167d3c9';
          sauce.createPublicLink(id, verifyLink(expected, done));
        });
      });
    });
  });
});

function verifySuccess(mock, done) {
  return function (err, data) {
    if (err) return done(new Error(err.error));
    mock.isDone().should.be.true;
    done();
  };
}

function verifyFailure(error, done) {
  return function (err, data) {
    if (!err) return done(new Error('Request succeeded'));
    err.should.deep.equal(error);
    done();
  };
}

function verifyLink(url, done) {
  return function (err, data) {
    if (err) return done(new Error(err));
    data.should.equal(url);
    done();
  };
}
