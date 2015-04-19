

    function F(){}

    /**
     * Do fn.apply on a constructor.
     */
    function ctorApply(ctor, args) {
        F.prototype = ctor.prototype;
        var instance = new F();
        ctor.apply(instance, args);
        return instance;
    }

    module.exports = ctorApply;


