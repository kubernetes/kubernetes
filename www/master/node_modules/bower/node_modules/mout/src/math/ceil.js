define(function(){
    /**
     * Round value up with a custom radix.
     */
    function ceil(val, step){
        step = Math.abs(step || 1);
        return Math.ceil(val / step) * step;
    }

    return ceil;
});
