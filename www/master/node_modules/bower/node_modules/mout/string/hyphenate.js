var toString = require('../lang/toString');
var slugify = require('./slugify');
var unCamelCase = require('./unCamelCase');
    /**
     * Replaces spaces with hyphens, split camelCase text, remove non-word chars, remove accents and convert to lower case.
     */
    function hyphenate(str){
        str = toString(str);
        str = unCamelCase(str);
        return slugify(str, "-");
    }

    module.exports = hyphenate;

