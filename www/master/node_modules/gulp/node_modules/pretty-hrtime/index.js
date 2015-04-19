/*jshint node:true */

"use strict";

var minimalDesc = ['h', 'min', 's', 'ms', 'μs', 'ns'];
var verboseDesc = ['hour', 'minute', 'second', 'millisecond', 'microsecond', 'nanosecond'];
var convert = [60*60, 60, 1, 1e6, 1e3, 1];

module.exports = function (source, opts) {
	var verbose, precise, i, spot, sourceAtStep, valAtStep, decimals, strAtStep, results;

	verbose = false;
	precise = false;
	if (opts) {
		verbose = opts.verbose || false;
		precise = opts.precise || false;
	}

	if (!Array.isArray(source) || source.length !== 2) {
		return '';
	}
	if (typeof source[0] !== 'number' || typeof source[1] !== 'number') {
		return '';
	}

	results = '';

	// foreach unit
	for (i = 0; i < 6; i++) {
		spot = i < 3 ? 0 : 1; // grabbing first or second spot in source array
		sourceAtStep = source[spot];
		if (i !== 3 && i !== 0) {
			sourceAtStep = sourceAtStep % convert[i-1]; // trim off previous portions
		}
		if (i === 2) {
			sourceAtStep += source[1]/1e9; // get partial seconds from other portion of the array
		}
		valAtStep = sourceAtStep / convert[i]; // val at this unit
		if (valAtStep >= 1) {
			if (verbose) {
				valAtStep = Math.floor(valAtStep); // deal in whole units, subsequent laps will get the decimal portion
			}
			if (!precise) {
				// don't fling too many decimals
				decimals = valAtStep >= 10 ? 0 : 2;
				strAtStep = valAtStep.toFixed(decimals);
			} else {
				strAtStep = valAtStep.toString();
			}
			if (strAtStep.indexOf('.') > -1 && strAtStep[strAtStep.length-1] === '0') {
				strAtStep = strAtStep.replace(/\.?0+$/,''); // remove trailing zeros
			}
			if (results) {
				results += ' '; // append space if we have a previous value
			}
			results += strAtStep; // append the value
			// append units
			if (verbose) {
				results += ' '+verboseDesc[i];
				if (strAtStep !== '1') {
					results += 's';
				}
			} else {
				results += ' '+minimalDesc[i];
			}
			if (!verbose) {
				break; // verbose gets as many groups as necessary, the rest get only one
			}
		}
	}

	return results;
};
