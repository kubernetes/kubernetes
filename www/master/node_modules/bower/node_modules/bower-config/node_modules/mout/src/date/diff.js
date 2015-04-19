define(['./totalDaysInMonth', './totalDaysInYear', '../time/convert'], function(totalDaysInMonth, totalDaysInYear, convert){

    /**
     * calculate the difference between dates (range)
     */
    function diff(start, end, unitName){
        // sort the dates to make it easier to process (specially year/month)
        if (start > end) {
            var swap = start;
            start = end;
            end = swap;
        }

        var output;

        if (unitName === 'month') {
            output = getMonthsDiff(start, end);
        } else if (unitName === 'year'){
            output = getYearsDiff(start, end);
        } else if (unitName != null) {
            if (unitName === 'day') {
                // ignore timezone difference because of daylight savings time
                start = toUtc(start);
                end = toUtc(end);
            }
            output = convert(end - start, 'ms', unitName);
        } else {
            output = end - start;
        }

        return output;
    }


    function toUtc(d){
        // we ignore timezone differences on purpose because of daylight
        // savings time, otherwise it would return fractional days/weeks even
        // if a full day elapsed. eg:
        // Wed Feb 12 2014 00:00:00 GMT-0200 (BRST)
        // Sun Feb 16 2014 00:00:00 GMT-0300 (BRT)
        // diff should be 4 days and not 4.041666666666667
        return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(),
                        d.getHours(), d.getMinutes(), d.getSeconds(),
                        d.getMilliseconds());
    }


    function getMonthsDiff(start, end){
        return getElapsedMonths(start, end) +
               getElapsedYears(start, end) * 12 +
               getFractionalMonth(start, end);
    }


    function getYearsDiff(start, end){
        var elapsedYears = getElapsedYears(start, end);
        return elapsedYears + getFractionalYear(start, end, elapsedYears);
    }


    function getElapsedMonths(start, end){
        var monthDiff = end.getMonth() - start.getMonth();
        if (monthDiff < 0) {
            monthDiff += 12;
        }
        // less than a full month
        if (start.getDate() > end.getDate()) {
            monthDiff -= 1;
        }
        return monthDiff;
    }


    function getElapsedYears(start, end){
        var yearDiff = end.getFullYear() - start.getFullYear();
        // less than a full year
        if (start.getMonth() > end.getMonth()) {
            yearDiff -= 1;
        }
        return yearDiff;
    }


    function getFractionalMonth(start, end){
        var fractionalDiff = 0;
        var startDay = start.getDate();
        var endDay = end.getDate();

        if (startDay !== endDay) {
            var startTotalDays = totalDaysInMonth(start);
            var endTotalDays = totalDaysInMonth(end);
            var totalDays;
            var daysElapsed;

            if (startDay > endDay) {
                // eg: Jan 29 - Feb 27 (29 days elapsed but not a full month)
                var baseDay = startTotalDays - startDay;
                daysElapsed = endDay + baseDay;
                // total days should be relative to 1st day of next month if
                // startDay > endTotalDays
                totalDays = (startDay > endTotalDays)?
                    endTotalDays + baseDay + 1 : startDay + baseDay;
            } else {
                // fractional is only based on endMonth eg: Jan 12 - Feb 18
                // (6 fractional days, 28 days until next full month)
                daysElapsed = endDay - startDay;
                totalDays = endTotalDays;
            }

            fractionalDiff = daysElapsed / totalDays;
        }

        return fractionalDiff;
    }


    function getFractionalYear(start, end, elapsedYears){
        var base = elapsedYears?
            new Date(end.getFullYear(), start.getMonth(), start.getDate()) :
            start;
        var elapsedDays = diff(base, end, 'day');
        return elapsedDays / totalDaysInYear(end);
    }


    return diff;

});
