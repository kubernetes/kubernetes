var Q = require('q');
var mout = require('mout');

var analytics = module.exports;

var insight;

var enableAnalytics = false;

// Insight takes long to load, and often causes problems
// in non-interactive environment, so we load it lazily
//
// Insight is used in two cases:
//
// 1. Read insight configuration (whether track user actions)
// 2. Track user actions (Tracker.track method)
//
// We don't want to instantiate Insight in non-interactive mode
// because it takes time to read config and configstore has concurrency issues:
//
// https://github.com/yeoman/configstore/issues/20
function ensureInsight () {
    if (!insight) {
        var Insight = require('insight');
        insight = new Insight({
            trackingCode: 'UA-43531210-1',
            pkg: require('../../package.json')
        });
    }
}

// Initializes the application-wide insight singleton and asks for the
// permission on the CLI during the first run.
//
// This method is called only from bin/bower. Programmatic API skips it.
analytics.setup = function setup (config) {
    var deferred = Q.defer();

    // No need for asking if analytics is set in bower config
    if (config.analytics === undefined) {
        ensureInsight();

        // For non-interactive call from bin/bower we disable analytics
        if (config.interactive) {
            if (insight.optOut !== undefined) {
                deferred.resolve(!insight.optOut);
            } else {
                insight.askPermission(null, function(err, optIn) {
                    // optIn callback param was exactly opposite before 0.4.3
                    // so we force at least insight@0.4.3 in package.json
                    deferred.resolve(optIn);
                });
            }
        } else {
            // no specified value, no stored value, and can't prompt for one
            // most likely CI environment; defaults to false to reduce data noise
            deferred.resolve(false);
        }
    } else {
        // use the specified value
        deferred.resolve(config.analytics);
    }

    return deferred.promise.then(function (enabled) {
        enableAnalytics = enabled;

        return enabled;
    });
};

var Tracker = analytics.Tracker = function Tracker(config) {
    function analyticsEnabled () {
        // Allow for overriding analytics default
        if (config && config.analytics !== undefined) {
            return config.analytics;
        }

        // TODO: let bower pass this variable from bin/bower instead closure
        return enableAnalytics;
    }

    if (analyticsEnabled()) {
        ensureInsight();
    } else {
        this.track = function noop () {};
        this.trackDecomposedEndpoints = function noop () {};
        this.trackPackages = function noop () {};
        this.trackNames = function noop () {};
    }
};

Tracker.prototype.track = function track() {
    insight.track.apply(insight, arguments);
};

Tracker.prototype.trackDecomposedEndpoints = function trackDecomposedEndpoints(command, endpoints) {
    endpoints.forEach(function (endpoint) {
        this.track(command, endpoint.source, endpoint.target);
    }.bind(this));
};

Tracker.prototype.trackPackages = function trackPackages(command, packages) {
    mout.object.forOwn(packages, function (package) {
        var meta = package.pkgMeta;
        this.track(command, meta.name, meta.version);
    }.bind(this));
};

Tracker.prototype.trackNames = function trackNames(command, names) {
    names.forEach(function (name) {
        this.track(command, name);
    }.bind(this));
};
