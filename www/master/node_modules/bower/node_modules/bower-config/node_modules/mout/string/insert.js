var clamp = require('../math/clamp');
var toString = require('../lang/toString');

    /**
     * Inserts a string at a given index.
     */
    function insert(string, index, partial){
        string = toString(string);

        if (index < 0) {
            index = string.length + index;
        }

        index = clamp(index, 0, string.length);

        return string.substr(0, index) + partial + string.substr(index);
    }

    module.exports = insert;


