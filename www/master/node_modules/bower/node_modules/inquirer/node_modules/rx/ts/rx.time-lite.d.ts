// DefinitelyTyped: partial

// This file contains common part of defintions for rx.time.d.ts and rx.lite.d.ts
// Do not include the file separately.

///<reference path="rx-lite.d.ts" />

declare module Rx {
	export interface TimeInterval<T> {
		value: T;
		interval: number;
	}

	export interface Timestamp<T> {
		value: T;
		timestamp: number;
	}

	export interface Observable<T> {
		delay(dueTime: Date, scheduler?: IScheduler): Observable<T>;
		delay(dueTime: number, scheduler?: IScheduler): Observable<T>;

		debounce(dueTime: number, scheduler?: IScheduler): Observable<T>;
		throttleWithTimeout(dueTime: number, scheduler?: IScheduler): Observable<T>;
		/**
		* @deprecated use #debounce or #throttleWithTimeout instead.
		*/
		throttle(dueTime: number, scheduler?: IScheduler): Observable<T>;

		timeInterval(scheduler?: IScheduler): Observable<TimeInterval<T>>;

		timestamp(scheduler?: IScheduler): Observable<Timestamp<T>>;

		sample(interval: number, scheduler?: IScheduler): Observable<T>;
		sample<TSample>(sampler: Observable<TSample>, scheduler?: IScheduler): Observable<T>;

		timeout(dueTime: Date, other?: Observable<T>, scheduler?: IScheduler): Observable<T>;
		timeout(dueTime: number, other?: Observable<T>, scheduler?: IScheduler): Observable<T>;
	}

	interface ObservableStatic {
		interval(period: number, scheduler?: IScheduler): Observable<number>;
		interval(dutTime: number, period: number, scheduler?: IScheduler): Observable<number>;
		timer(dueTime: number, period: number, scheduler?: IScheduler): Observable<number>;
		timer(dueTime: number, scheduler?: IScheduler): Observable<number>;
	}
}
