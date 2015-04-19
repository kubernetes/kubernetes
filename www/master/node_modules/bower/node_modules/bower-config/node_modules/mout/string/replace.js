var toString = require('../lang/toString');
var toArray = require('../lang/toArray');

    /**
     * Replace string(s) with the replacement(s) in the source.
     */
    function replace(str, search, replacements) {
        str = toString(str);
        search = toArray(search);
        replacements = toArray(replacements);

        var searchLength = search.length,
            replacementsLength = replacements.length;

        if (replacementsLength !== 1 && searchLength !== replacementsLength) {
            throw new Error('Unequal number of searches and replacements');
        }

        var i = -1;
        while (++i < searchLength) {
            // Use the first replacement for all searches if only one
            // replacement is provided
            str = str.replace(
                search[i],
                replacements[(replacementsLength === 1) ? 0 : i]);
        }

        return str;
    }

    module.exports = replace;


