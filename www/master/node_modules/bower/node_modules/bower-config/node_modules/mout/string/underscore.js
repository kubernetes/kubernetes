var toString = require('../lang/toString');
var slugify = require('./slugify');
var unCamelCase = require('./unCamelCase');
    /**
     * Replaces spaces with underscores, split camelCase text, remove non-word chars, remove accents and convert to lower case.
     */
    function underscore(str){
        str = toString(str);
        str = unCamelCase(str);
        return slugify(str, "_");
    }
    module.exports = underscore;

