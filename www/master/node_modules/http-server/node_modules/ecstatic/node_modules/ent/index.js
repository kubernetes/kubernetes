var punycode = require('punycode');
var entities = require('./entities.json');

var revEntities = {};
for (var key in entities) {
    var e = entities[key];
    var s = typeof e === 'number' ? String.fromCharCode(e) : e;
    revEntities[s] = key;
}

exports.encode = function (str) {
    if (typeof str !== 'string') {
        throw new TypeError('Expected a String');
    }
    
    return str.split('').map(function (c) {
        var e = revEntities[c];
        var cc = c.charCodeAt(0);
        if (e) {
            return '&' + (e.match(/;$/) ? e : e + ';');
        }
        else if (c.match(/\s/)) {
            return c;
        }
        else if (cc < 32 || cc >= 127) {
            return '&#' + cc + ';';
        }
        else {
            return c;
        }
    }).join('');
};

exports.decode = function (str) {
    if (typeof str !== 'string') {
        throw new TypeError('Expected a String');
    }
    
    return str
        .replace(/&#(\d+);?/g, function (_, code) {
            return punycode.ucs2.encode([code]);
        })
        .replace(/&#[xX]([A-Fa-f0-9]+);?/g, function (_, hex) {
            return punycode.ucs2.encode([parseInt(hex, 16)]);
        })
        .replace(/&([^;\W]+;?)/g, function (m, e) {
            var ee = e.replace(/;$/, '');
            var target = entities[e]
                || (e.match(/;$/) && entities[ee])
            ;
            
            if (typeof target === 'number') {
                return punycode.ucs2.encode([target]);
            }
            else if (typeof target === 'string') {
                return target;
            }
            else {
                return m;
            }
        })
    ;
};
