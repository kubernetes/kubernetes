function decompose(endpoint) {
    // Note that we allow spaces in targets and sources but they are trimmed
    var regExp = /^(?:([\w\-]|(?:[\w\.\-]+[\w\-])?)=)?([^\|#]+)(?:#(.*))?$/;
    var matches = endpoint.match(regExp);
    var target;
    var error;

    if (!matches) {
        error = new Error('Invalid endpoint: ' + endpoint);
        error.code = 'EINVEND';
        throw error;
    }

    target = trim(matches[3]);

    return {
        name: trim(matches[1]),
        source: trim(matches[2]),
        target: isWildcard(target) ? '*' : target
    };
}

function compose(decEndpoint) {
    var name = trim(decEndpoint.name);
    var source = trim(decEndpoint.source);
    var target = trim(decEndpoint.target);
    var composed = '';

    if (name) {
        composed += name + '=';
    }

    composed += source;

    if (!isWildcard(target)) {
        composed += '#' + target;
    }

    return composed;
}

function json2decomposed(key, value) {
    var endpoint;
    var split;
    var error;

    key = trim(key);
    value = trim(value);

    if (!key) {
        error = new Error('The key must be specified');
        error.code = 'EINVEND';
        throw error;
    }

    endpoint = key + '=';
    split = value.split('#').map(trim);

    // If # was found, the source was specified
    if (split.length > 1) {
        endpoint += (split[0] || key) + '#' + split[1];
    // Check if value looks like a source
    } else if (isSource(value)) {
        endpoint += value + '#*';
    // Otherwise use the key as the source
    } else {
        endpoint += key + '#' + split[0];
    }

    return decompose(endpoint);
}

function decomposed2json(decEndpoint) {
    var error;
    var name = trim(decEndpoint.name);
    var source = trim(decEndpoint.source);
    var target = trim(decEndpoint.target);
    var value = '';
    var ret = {};

    if (!name) {
        error = new Error('Decomposed endpoint must have a name');
        error.code = 'EINVEND';
        throw error;
    }

    // Add source only if different than the name
    if  (source !== name) {
        value += source;
    }

    // If value is empty, we append the target always
    if (!value) {
        if (isWildcard(target)) {
            value += '*';
        } else {
            if (target.indexOf('/') !== -1) {
                value += '#' + target;
            } else {
                value += target;
            }
        }
    // Otherwise append only if not a wildcard or source does not look like a source
    } else if (!isWildcard(target) || !isSource(source)) {
        value += '#' + (target || '*');
    }

    ret[name] = value;

    return ret;
}

function trim(str) {
    return str ? str.trim() : '';
}

function isWildcard(target) {
    return !target || target === '*' || target === 'latest';
}

function isSource(value) {
    return (/[\/\\@]/).test(value);
}

module.exports.decompose = decompose;
module.exports.compose = compose;
module.exports.json2decomposed = json2decomposed;
module.exports.decomposed2json = decomposed2json;
