// Based on https://gist.github.com/mathiasbynens/6334847 by @mathias

var regenerate = require('regenerate');

// Which Unicode version should be used?
var version = '7.0.0'; // note: also update `package.json` when this changes

// Shorthand function
var get = function(what) {
    return require('unicode-' + version + '/' + what + '/code-points');
};

// Unicode categories needed to construct the ES5 regex
var Lu = get('categories/Lu');
var Ll = get('categories/Ll');
var Lt = get('categories/Lt');
var Lm = get('categories/Lm');
var Lo = get('categories/Lo');
var Nl = get('categories/Nl');
var Mn = get('categories/Mn');
var Mc = get('categories/Mc');
var Nd = get('categories/Nd');
var Pc = get('categories/Pc');

var generateES5Regex = function() { // ES 5.1
    // http://mathiasbynens.be/notes/javascript-identifiers#valid-identifier-names
    var identifierStart = regenerate('$', '_')
        .add(Lu, Ll, Lt, Lm, Lo, Nl)
        .removeRange(0x010000, 0x10FFFF) // remove astral symbols
        .removeRange(0x0, 0x7F); // remove ASCII symbols (Acorn-specific)
    var identifierPart = regenerate('\u200C', '\u200D', Mn, Mc, Nd, Pc)
        .removeRange(0x010000, 0x10FFFF) // remove astral symbols
        .remove(identifierStart) // (Acorn-specific)
        .removeRange(0x0, 0x7F); // remove ASCII symbols (Acorn-specific)
    return {
        'NonAsciiIdentifierStart': identifierStart.toString(),
        'NonAsciiIdentifierPart': identifierPart.toString()
    };
};

var result = generateES5Regex();
console.log(
    '// ECMAScript 5.1/Unicode v%s `nonASCIIidentifierStart`:\n\n%s\n',
    version,
    result.NonAsciiIdentifierStart
);
console.log(
    '// ECMAScript 5.1/Unicode v%s `nonASCIIidentifier`:\n\n%s',
    version,
    result.NonAsciiIdentifierPart
);
