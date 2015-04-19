define(function(){
    /**
    * Check if value is close to target.
    */
    function isNear(val, target, threshold){
        return (Math.abs(val - target) <= threshold);
    }
    return isNear;
});
