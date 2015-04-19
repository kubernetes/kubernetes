// DefinitelyTyped: partial

// This file contains common part of defintions for rx.binding.d.ts and rx.lite.d.ts
// Do not include the file separately.

///<reference path="rx-lite.d.ts"/>

declare module Rx {
	export interface BehaviorSubject<T> extends Subject<T> {
		getValue(): T;
	}

	interface BehaviorSubjectStatic {
		new <T>(initialValue: T): BehaviorSubject<T>;
	}

	export var BehaviorSubject: BehaviorSubjectStatic;

	export interface ReplaySubject<T> extends Subject<T> {
	}

	interface ReplaySubjectStatic {
		new <T>(bufferSize?: number, window?: number, scheduler?: IScheduler): ReplaySubject<T>;
	}

	export var ReplaySubject: ReplaySubjectStatic;

	interface ConnectableObservable<T> extends Observable<T> {
		connect(): IDisposable;
		refCount(): Observable<T>;
    }

    interface ConnectableObservableStatic {
        new <T>(): ConnectableObservable<T>;
	}

	export var ConnectableObservable: ConnectableObservableStatic;

	export interface Observable<T> {
		multicast(subject: Observable<T>): ConnectableObservable<T>;
		multicast<TResult>(subjectSelector: () => ISubject<T>, selector: (source: ConnectableObservable<T>) => Observable<T>): Observable<T>;
		publish(): ConnectableObservable<T>;
		publish<TResult>(selector: (source: ConnectableObservable<T>) => Observable<TResult>): Observable<TResult>;
		/**
		* Returns an observable sequence that shares a single subscription to the underlying sequence.
		* This operator is a specialization of publish which creates a subscription when the number of observers goes from zero to one, then shares that subscription with all subsequent observers until the number of observers returns to zero, at which point the subscription is disposed.
		*
		* @example
		* var res = source.share();
		*
		* @returns An observable sequence that contains the elements of a sequence produced by multicasting the source sequence.
		*/
		share(): Observable<T>;
		publishLast(): ConnectableObservable<T>;
		publishLast<TResult>(selector: (source: ConnectableObservable<T>) => Observable<TResult>): Observable<TResult>;
		publishValue(initialValue: T): ConnectableObservable<T>;
		publishValue<TResult>(selector: (source: ConnectableObservable<T>) => Observable<TResult>, initialValue: T): Observable<TResult>;
		/**
		* Returns an observable sequence that shares a single subscription to the underlying sequence and starts with an initialValue.
		* This operator is a specialization of publishValue which creates a subscription when the number of observers goes from zero to one, then shares that subscription with all subsequent observers until the number of observers returns to zero, at which point the subscription is disposed.
		*
		* @example
		* var res = source.shareValue(42);
		*
		* @param initialValue Initial value received by observers upon subscription.
		* @returns An observable sequence that contains the elements of a sequence produced by multicasting the source sequence.
		*/
		shareValue(initialValue: T): Observable<T>;
		replay(selector?: boolean, bufferSize?: number, window?: number, scheduler?: IScheduler): ConnectableObservable<T>;	// hack to catch first omitted parameter
		replay(selector: (source: ConnectableObservable<T>) => Observable<T>, bufferSize?: number, window?: number, scheduler?: IScheduler): Observable<T>;
		shareReplay(bufferSize?: number, window?: number, scheduler?: IScheduler): Observable<T>;
	}
}
