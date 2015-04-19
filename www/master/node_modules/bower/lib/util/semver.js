var semver = require('semver');
var mout = require('mout');

function maxSatisfying(versions, range, strictMatch) {
    var version;
    var filteredVersions;

    // Filter only valid versions, since semver.maxSatisfying() throws an error
    versions = versions.filter(function (version) {
        return semver.valid(version);
    });

    // Exact version & range match
    if (semver.valid(range)) {
        version = mout.array.find(versions, function (version) {
            return version === range;
        });
        if (version) {
            return version;
        }
    }

    range = typeof range === 'string' ? range.trim() : range;

    // When strict match is enabled give priority to non-pre-releases
    // We do this by filtering every pre-release version
    if (strictMatch) {
        filteredVersions = versions.map(function (version) {
            return !isPreRelease(version) ? version : null;
        });

        version = semver.maxSatisfying(filteredVersions, range);
        if (version) {
            return version;
        }
    }

    // Fallback to regular semver max satisfies
    return semver.maxSatisfying(versions, range);
}

function maxSatisfyingIndex(versions, range, strictMatch) {
    var version = maxSatisfying(versions, range, strictMatch);

    if (!version) {
        return -1;
    }

    return versions.indexOf(version);
}

function clean(version) {
    var parsed = semver.parse(version);

    if (!parsed) {
        return null;
    }

    // Keep builds!
    return parsed.version + (parsed.build.length ? '+' + parsed.build.join('.') : '');
}

function isPreRelease(version) {
    var parsed = semver.parse(version);
    return parsed && parsed.prerelease && parsed.prerelease.length;
}

// Export a semver like object but with our custom functions
mout.object.mixIn(module.exports, semver, {
    maxSatisfying: maxSatisfying,
    maxSatisfyingIndex: maxSatisfyingIndex,
    clean: clean,
    valid: clean
});
