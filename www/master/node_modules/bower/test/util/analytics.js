var expect = require('expect.js');
var proxyquire = require('proxyquire');
var object = require('mout').object;

describe('analytics', function () {

    var mockAnalytics = function(stubs, promptResponse) {
        return proxyquire('../../lib/util/analytics', {
            insight: function () {
                return object.merge(stubs || {}, {
                    askPermission: function (message, callback) {
                        callback(undefined, promptResponse);
                    }
                });
            }
        });
    };

    describe('#setup', function () {
        it('leaves analytics enabled if provided', function () {
            return mockAnalytics()
                .setup({ analytics: true })
                .then(function (enabled) {
                    expect(enabled).to.be(true);
                });
        });

        it('leaves analytics disabled if provided', function () {
            return mockAnalytics()
                .setup({ analytics: false })
                .then(function (enabled) {
                    expect(enabled).to.be(false);
                });
        });

        it('disables analytics for non-interactive mode', function () {
            return mockAnalytics()
                .setup({ interactive: false })
                .then(function (enabled) {
                    expect(enabled).to.be(false);
                });
        });

        it('disables if insight.optOut is true and interactive', function () {
            return mockAnalytics({ optOut: true })
                .setup({ interactive: true })
                .then(function (enabled) {
                    expect(enabled).to.be(false);
                });
        });

        it('enables if insight.optOut is false and interactive', function () {
            return mockAnalytics({ optOut: false })
                .setup({ interactive: true })
                .then(function (enabled) {
                    expect(enabled).to.be(true);
                });
        });

        it('disables if insight.optOut is false and non-interactive', function () {
            return mockAnalytics({ optOut: false })
                .setup({ interactive: false })
                .then(function (enabled) {
                    expect(enabled).to.be(false);
                });
        });

        it('enables if interactive insights return true from prompt', function () {
            return mockAnalytics({ optOut: undefined }, true)
                .setup({ interactive: true })
                .then(function (enabled) {
                    expect(enabled).to.be(true);
                });
        });

        it('disables if interactive insights return false from prompt', function () {
            return mockAnalytics({ optOut: undefined }, false)
                .setup({ interactive: true })
                .then(function (enabled) {
                    expect(enabled).to.be(false);
                });
        });
    });

    describe('Tracker', function (next) {
        it('tracks if analytics = true', function(next) {
            var analytics = mockAnalytics({
                track: function (arg) {
                    expect(arg).to.be('foo');
                    next();
                }
            });

            new analytics.Tracker({ analytics: true }).track('foo');
        });

        it('does not track if analytics = false', function () {
            var analytics = mockAnalytics({
                track: function (arg) {
                    throw new Error();
                }
            });

            expect(function () {
                new analytics.Tracker({ analytics: false }).track('foo');
            }).to.not.throwError();
        });

        it('tracks if analytics = undefined and setup returns true', function(next) {
            var analytics = mockAnalytics({
                track: function (arg) {
                    expect(arg).to.be('foo');
                    next();
                }
            });

            analytics
                .setup({ analytics: true })
                .then(function () {
                    new analytics.Tracker({}).track('foo');
                });
        });
    });
});
