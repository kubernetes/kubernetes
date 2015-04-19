define(function () {

    /**
     * convert time into another unit
     */
    function convert(val, sourceUnitName, destinationUnitName){
        destinationUnitName = destinationUnitName || 'ms';
        return (val * getUnit(sourceUnitName)) / getUnit(destinationUnitName);
    }


    //TODO: maybe extract to a separate module
    function getUnit(unitName){
        switch(unitName){
            case 'ms':
            case 'millisecond':
                return 1;
            case 's':
            case 'second':
                 return 1000;
            case 'm':
            case 'minute':
                 return 60000;
            case 'h':
            case 'hour':
                 return 3600000;
            case 'd':
            case 'day':
                 return 86400000;
            case 'w':
            case 'week':
                 return 604800000;
            default:
                throw new Error('"'+ unitName + '" is not a valid unit');
        }
    }


    return convert;

});
