var toArray = require('./toArray');
var find = require('../array/find');

    /**
     * Return first non void argument
     */
    function defaults(var_args){
        return find(toArray(arguments), nonVoid);
    }

    function nonVoid(val){
        return val != null;
    }

    module.exports = defaults;


