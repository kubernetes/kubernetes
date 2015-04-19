
    /**
    * Gets normalized ratio of value inside range.
    */
    function norm(val, min, max){
        return (val - min) / (max - min);
    }
    module.exports = norm;

