define(function () {

    // Reference to the global context (works on ES3 and ES5-strict mode)
    //jshint -W061, -W064
    return Function('return this')();

});
