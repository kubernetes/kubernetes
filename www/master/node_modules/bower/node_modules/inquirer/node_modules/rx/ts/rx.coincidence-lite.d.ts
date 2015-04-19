// DefinitelyTyped: partial

// This file contains common part of defintions for rx.time.d.ts and rx.lite.d.ts
// Do not include the file separately.

///<reference path="rx-lite.d.ts" />

declare module Rx {

	interface Observable<T> {
		/**
		* Returns a new observable that triggers on the second and subsequent triggerings of the input observable.
		* The Nth triggering of the input observable passes the arguments from the N-1th and Nth triggering as a pair.
		* The argument passed to the N-1th triggering is held in hidden internal state until the Nth triggering occurs.
		* @returns An observable that triggers on successive pairs of observations from the input observable as an array.
		*/
		pairwise(): Observable<T[]>;

		/**
		* Returns two observables which partition the observations of the source by the given function.
		* The first will trigger observations for those values for which the predicate returns true.
		* The second will trigger observations for those values where the predicate returns false.
		* The predicate is executed once for each subscribed observer.
		* Both also propagate all error observations arising from the source and each completes
		* when the source completes.
		* @param predicate
		*    The function to determine which output Observable will trigger a particular observation.
		* @returns
		*    An array of observables. The first triggers when the predicate returns true,
		*    and the second triggers when the predicate returns false.
		*/
		partition(predicate: (value: T, index: number, source: Observable<T>) => boolean, thisArg: any): Observable<T>[];
	}
}
