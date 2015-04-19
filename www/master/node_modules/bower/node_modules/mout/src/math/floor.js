define(function(){
    /**
    * Floor value to full steps.
    */
    function floor(val, step){
        step = Math.abs(step || 1);
        return Math.floor(val / step) * step;
    }
    return floor;
});
