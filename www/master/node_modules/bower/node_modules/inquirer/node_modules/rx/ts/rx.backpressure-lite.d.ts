// DefinitelyTyped: partial

// This file contains common part of defintions for rx.backpressure.d.ts and rx.lite.d.ts
// Do not include the file separately.

///<reference path="rx-lite.d.ts"/>

declare module Rx {
	export interface Observable<T> {
		/**
		* Pauses the underlying observable sequence based upon the observable sequence which yields true/false.
		* @example
		* var pauser = new Rx.Subject();
		* var source = Rx.Observable.interval(100).pausable(pauser);
		* @param pauser The observable sequence used to pause the underlying sequence.
		* @returns The observable sequence which is paused based upon the pauser.
		*/
		pausable(pauser?: Observable<boolean>): PausableObservable<T>;

		/**
		* Pauses the underlying observable sequence based upon the observable sequence which yields true/false,
		* and yields the values that were buffered while paused.
		* @example
		* var pauser = new Rx.Subject();
		* var source = Rx.Observable.interval(100).pausableBuffered(pauser);
		* @param pauser The observable sequence used to pause the underlying sequence.
		* @returns The observable sequence which is paused based upon the pauser.
		*/
		pausableBuffered(pauser?: Observable<boolean>): PausableObservable<T>;

		/**
		* Attaches a controller to the observable sequence with the ability to queue.
		* @example
		* var source = Rx.Observable.interval(100).controlled();
		* source.request(3); // Reads 3 values
		*/
		controlled(enableQueue?: boolean): ControlledObservable<T>;
	}

	export interface ControlledObservable<T> extends Observable<T> {
		request(numberOfItems?: number): IDisposable;
	}

	export interface PausableObservable<T> extends Observable<T> {
		pause(): void;
		resume(): void;
	}
}
