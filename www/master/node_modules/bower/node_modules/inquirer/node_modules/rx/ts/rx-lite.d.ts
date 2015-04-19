// DefinitelyTyped: partial

// This file contains common part of defintions for rx.d.ts and rx.lite.d.ts
// Do not include the file separately.

declare module Rx {
	export module internals {
		function isEqual(left: any, right: any): boolean;
		function addRef<T>(xs: Observable<T>, r: { getDisposable(): IDisposable; }): Observable<T>;

		// Priority Queue for Scheduling
		export class PriorityQueue<TTime> {
			constructor(capacity: number);

			length: number;

			isHigherPriority(left: number, right: number): boolean;
			percolate(index: number): void;
			heapify(index: number): void;
			peek(): ScheduledItem<TTime>;
			removeAt(index: number): void;
			dequeue(): ScheduledItem<TTime>;
			enqueue(item: ScheduledItem<TTime>): void;
			remove(item: ScheduledItem<TTime>): boolean;

			static count: number;
		}

		export class ScheduledItem<TTime> {
			constructor(scheduler: IScheduler, state: any, action: (scheduler: IScheduler, state: any) => IDisposable, dueTime: TTime, comparer?: (x: TTime, y: TTime) => number);

			scheduler: IScheduler;
			state: TTime;
			action: (scheduler: IScheduler, state: any) => IDisposable;
			dueTime: TTime;
			comparer: (x: TTime, y: TTime) => number;
			disposable: SingleAssignmentDisposable;

			invoke(): void;
			compareTo(other: ScheduledItem<TTime>): number;
			isCancelled(): boolean;
			invokeCore(): IDisposable;
		}
	}

	export module config {
		export var Promise: { new <T>(resolver: (resolvePromise: (value: T) => void, rejectPromise: (reason: any) => void) => void): IPromise<T>; };
	}

	export module helpers {
		function noop(): void;
		function notDefined(value: any): boolean;
		function isScheduler(value: any): boolean;
		function identity<T>(value: T): T;
		function defaultNow(): number;
		function defaultComparer(left: any, right: any): boolean;
		function defaultSubComparer(left: any, right: any): number;
		function defaultKeySerializer(key: any): string;
		function defaultError(err: any): void;
		function isPromise(p: any): boolean;
		function asArray<T>(...args: T[]): T[];
		function not(value: any): boolean;
		function isFunction(value: any): boolean;
	}

	export interface IDisposable {
		dispose(): void;
	}

	export class CompositeDisposable implements IDisposable {
		constructor (...disposables: IDisposable[]);
		constructor (disposables: IDisposable[]);

		isDisposed: boolean;
		length: number;

		dispose(): void;
		add(item: IDisposable): void;
		remove(item: IDisposable): boolean;
		toArray(): IDisposable[];
	}

	export class Disposable implements IDisposable {
		constructor(action: () => void);

		static create(action: () => void): IDisposable;
		static empty: IDisposable;

		dispose(): void;
	}

	// Single assignment
	export class SingleAssignmentDisposable implements IDisposable {
		constructor();

		isDisposed: boolean;
		current: IDisposable;

		dispose(): void ;
		getDisposable(): IDisposable;
		setDisposable(value: IDisposable): void ;
	}

	// SerialDisposable it's an alias of SingleAssignmentDisposable
	export class SerialDisposable extends SingleAssignmentDisposable {
		constructor();
	}

	export class RefCountDisposable implements IDisposable {
		constructor(disposable: IDisposable);

		dispose(): void;

		isDisposed: boolean;
		getDisposable(): IDisposable;
	}

	export interface IScheduler {
		now(): number;

		schedule(action: () => void): IDisposable;
		scheduleWithState<TState>(state: TState, action: (scheduler: IScheduler, state: TState) => IDisposable): IDisposable;
		scheduleWithAbsolute(dueTime: number, action: () => void): IDisposable;
		scheduleWithAbsoluteAndState<TState>(state: TState, dueTime: number, action: (scheduler: IScheduler, state: TState) =>IDisposable): IDisposable;
		scheduleWithRelative(dueTime: number, action: () => void): IDisposable;
		scheduleWithRelativeAndState<TState>(state: TState, dueTime: number, action: (scheduler: IScheduler, state: TState) =>IDisposable): IDisposable;

		scheduleRecursive(action: (action: () =>void ) =>void ): IDisposable;
		scheduleRecursiveWithState<TState>(state: TState, action: (state: TState, action: (state: TState) =>void ) =>void ): IDisposable;
		scheduleRecursiveWithAbsolute(dueTime: number, action: (action: (dueTime: number) => void) => void): IDisposable;
		scheduleRecursiveWithAbsoluteAndState<TState>(state: TState, dueTime: number, action: (state: TState, action: (state: TState, dueTime: number) => void) => void): IDisposable;
		scheduleRecursiveWithRelative(dueTime: number, action: (action: (dueTime: number) =>void ) =>void ): IDisposable;
		scheduleRecursiveWithRelativeAndState<TState>(state: TState, dueTime: number, action: (state: TState, action: (state: TState, dueTime: number) =>void ) =>void ): IDisposable;

		schedulePeriodic(period: number, action: () => void): IDisposable;
		schedulePeriodicWithState<TState>(state: TState, period: number, action: (state: TState) => TState): IDisposable;
	}

	export interface Scheduler extends IScheduler {
	}

	export interface SchedulerStatic {
		new (
			now: () => number,
			schedule: (state: any, action: (scheduler: IScheduler, state: any) => IDisposable) => IDisposable,
			scheduleRelative: (state: any, dueTime: number, action: (scheduler: IScheduler, state: any) => IDisposable) => IDisposable,
			scheduleAbsolute: (state: any, dueTime: number, action: (scheduler: IScheduler, state: any) => IDisposable) => IDisposable): Scheduler;

		normalize(timeSpan: number): number;

		immediate: IScheduler;
		currentThread: ICurrentThreadScheduler;
		timeout: IScheduler;
	}

	export var Scheduler: SchedulerStatic;

	// Current Thread IScheduler
	interface ICurrentThreadScheduler extends IScheduler {
		scheduleRequired(): boolean;
	}

	// Notifications
	export class Notification<T> {
		accept(observer: IObserver<T>): void;
		accept<TResult>(onNext: (value: T) => TResult, onError?: (exception: any) => TResult, onCompleted?: () => TResult): TResult;
		toObservable(scheduler?: IScheduler): Observable<T>;
		hasValue: boolean;
		equals(other: Notification<T>): boolean;
		kind: string;
		value: T;
		exception: any;

		static createOnNext<T>(value: T): Notification<T>;
		static createOnError<T>(exception: any): Notification<T>;
		static createOnCompleted<T>(): Notification<T>;
	}

	/**
	 * Promise A+
	 */
	export interface IPromise<T> {
		then<R>(onFulfilled: (value: T) => IPromise<R>, onRejected: (reason: any) => IPromise<R>): IPromise<R>;
		then<R>(onFulfilled: (value: T) => IPromise<R>, onRejected?: (reason: any) => R): IPromise<R>;
		then<R>(onFulfilled: (value: T) => R, onRejected: (reason: any) => IPromise<R>): IPromise<R>;
		then<R>(onFulfilled?: (value: T) => R, onRejected?: (reason: any) => R): IPromise<R>;
	}

	// Observer
	export interface IObserver<T> {
		onNext(value: T): void;
		onError(exception: any): void;
		onCompleted(): void;
	}

	export interface Observer<T> extends IObserver<T> {
		toNotifier(): (notification: Notification<T>) => void;
		asObserver(): Observer<T>;
	}

	interface ObserverStatic {
		create<T>(onNext?: (value: T) => void, onError?: (exception: any) => void, onCompleted?: () => void): Observer<T>;
		fromNotifier<T>(handler: (notification: Notification<T>, thisArg?: any) => void): Observer<T>;
	}

	export var Observer: ObserverStatic;

	export interface IObservable<T> {
		subscribe(observer: Observer<T>): IDisposable;
		subscribe(onNext?: (value: T) => void, onError?: (exception: any) => void, onCompleted?: () => void): IDisposable;

		subscribeOnNext(onNext: (value: T) => void, thisArg?: any): IDisposable;
		subscribeOnError(onError: (exception: any) => void, thisArg?: any): IDisposable;
		subscribeOnCompleted(onCompleted: () => void, thisArg?: any): IDisposable;
	}

	export interface Observable<T> extends IObservable<T> {
		forEach(onNext?: (value: T) => void, onError?: (exception: any) => void, onCompleted?: () => void): IDisposable;	// alias for subscribe
		toArray(): Observable<T[]>;

		catch(handler: (exception: any) => Observable<T>): Observable<T>;
		catchException(handler: (exception: any) => Observable<T>): Observable<T>;	// alias for catch
		catch(handler: (exception: any) => IPromise<T>): Observable<T>;
		catchException(handler: (exception: any) => IPromise<T>): Observable<T>;	// alias for catch
		catch(second: Observable<T>): Observable<T>;
		catchException(second: Observable<T>): Observable<T>;	// alias for catch
		combineLatest<T2, TResult>(second: Observable<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		combineLatest<T2, TResult>(second: IPromise<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		combineLatest<T2, T3, TResult>(second: Observable<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T2, T3, TResult>(second: Observable<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T2, T3, TResult>(second: IPromise<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T2, T3, TResult>(second: IPromise<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, TResult>(second: Observable<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, TResult>(second: Observable<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, TResult>(second: IPromise<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, TResult>(second: IPromise<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, TResult>(second: IPromise<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, TResult>(second: IPromise<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T2, T3, T4, T5, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, fifth: Observable<T5>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4, v5: T5) => TResult): Observable<TResult>;
		combineLatest<TOther, TResult>(souces: Observable<TOther>[], resultSelector: (firstValue: T, ...otherValues: TOther[]) => TResult): Observable<TResult>;
		combineLatest<TOther, TResult>(souces: IPromise<TOther>[], resultSelector: (firstValue: T, ...otherValues: TOther[]) => TResult): Observable<TResult>;
		withLatestFrom<T2, TResult>(second: Observable<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		withLatestFrom<T2, TResult>(second: IPromise<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, TResult>(second: Observable<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, TResult>(second: Observable<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, TResult>(second: IPromise<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, TResult>(second: IPromise<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, TResult>(second: Observable<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, TResult>(second: Observable<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, TResult>(second: IPromise<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, TResult>(second: IPromise<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, TResult>(second: IPromise<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, TResult>(second: IPromise<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T2, T3, T4, T5, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, fifth: Observable<T5>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4, v5: T5) => TResult): Observable<TResult>;
		withLatestFrom<TOther, TResult>(souces: Observable<TOther>[], resultSelector: (firstValue: T, ...otherValues: TOther[]) => TResult): Observable<TResult>;
		withLatestFrom<TOther, TResult>(souces: IPromise<TOther>[], resultSelector: (firstValue: T, ...otherValues: TOther[]) => TResult): Observable<TResult>;
		concat(...sources: Observable<T>[]): Observable<T>;
		concat(...sources: IPromise<T>[]): Observable<T>;
		concat(sources: Observable<T>[]): Observable<T>;
		concat(sources: IPromise<T>[]): Observable<T>;
		concatAll(): T;
		concatObservable(): T;	// alias for concatAll
		concatMap<T2, R>(selector: (value: T, index: number) => Observable<T2>, resultSelector: (value1: T, value2: T2, index: number) => R): Observable<R>;	// alias for selectConcat
		concatMap<T2, R>(selector: (value: T, index: number) => IPromise<T2>, resultSelector: (value1: T, value2: T2, index: number) => R): Observable<R>;	// alias for selectConcat
		concatMap<R>(selector: (value: T, index: number) => Observable<R>): Observable<R>;	// alias for selectConcat
		concatMap<R>(selector: (value: T, index: number) => IPromise<R>): Observable<R>;	// alias for selectConcat
		concatMap<R>(sequence: Observable<R>): Observable<R>;	// alias for selectConcat
		merge(maxConcurrent: number): T;
		merge(other: Observable<T>): Observable<T>;
		merge(other: IPromise<T>): Observable<T>;
		mergeAll(): T;
		mergeObservable(): T;	// alias for mergeAll
		skipUntil<T2>(other: Observable<T2>): Observable<T>;
		skipUntil<T2>(other: IPromise<T2>): Observable<T>;
		switch(): T;
		switchLatest(): T;	// alias for switch
		takeUntil<T2>(other: Observable<T2>): Observable<T>;
		takeUntil<T2>(other: IPromise<T2>): Observable<T>;
		zip<T2, TResult>(second: Observable<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		zip<T2, TResult>(second: IPromise<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		zip<T2, T3, TResult>(second: Observable<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		zip<T2, T3, TResult>(second: Observable<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		zip<T2, T3, TResult>(second: IPromise<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		zip<T2, T3, TResult>(second: IPromise<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		zip<T2, T3, T4, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		zip<T2, T3, T4, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		zip<T2, T3, T4, TResult>(second: Observable<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		zip<T2, T3, T4, TResult>(second: Observable<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		zip<T2, T3, T4, TResult>(second: IPromise<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		zip<T2, T3, T4, TResult>(second: IPromise<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		zip<T2, T3, T4, TResult>(second: IPromise<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		zip<T2, T3, T4, TResult>(second: IPromise<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		zip<T2, T3, T4, T5, TResult>(second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, fifth: Observable<T5>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4, v5: T5) => TResult): Observable<TResult>;
		zip<TOther, TResult>(second: Observable<TOther>[], resultSelector: (left: T, ...right: TOther[]) => TResult): Observable<TResult>;
		zip<TOther, TResult>(second: IPromise<TOther>[], resultSelector: (left: T, ...right: TOther[]) => TResult): Observable<TResult>;

		asObservable(): Observable<T>;
		dematerialize<TOrigin>(): Observable<TOrigin>;
		distinctUntilChanged(skipParameter: boolean, comparer: (x: T, y: T) => boolean): Observable<T>;
		distinctUntilChanged<TValue>(keySelector?: (value: T) => TValue, comparer?: (x: TValue, y: TValue) => boolean): Observable<T>;
		do(observer: Observer<T>): Observable<T>;
		doAction(observer: Observer<T>): Observable<T>;	// alias for do
		tap(observer: Observer<T>): Observable<T>;	// alias for do
		do(onNext?: (value: T) => void, onError?: (exception: any) => void, onCompleted?: () => void): Observable<T>;
		doAction(onNext?: (value: T) => void, onError?: (exception: any) => void, onCompleted?: () => void): Observable<T>;	// alias for do
		tap(onNext?: (value: T) => void, onError?: (exception: any) => void, onCompleted?: () => void): Observable<T>;	// alias for do
		
		doOnNext(onNext: (value: T) => void, thisArg?: any): Observable<T>;
		doOnError(onError: (exception: any) => void, thisArg?: any): Observable<T>;
		doOnCompleted(onCompleted: () => void, thisArg?: any): Observable<T>;
		tapOnNext(onNext: (value: T) => void, thisArg?: any): Observable<T>;
		tapOnError(onError: (exception: any) => void, thisArg?: any): Observable<T>;
		tapOnCompleted(onCompleted: () => void, thisArg?: any): Observable<T>;

		finally(action: () => void): Observable<T>;
		finallyAction(action: () => void): Observable<T>;	// alias for finally
		ignoreElements(): Observable<T>;
		materialize(): Observable<Notification<T>>;
		repeat(repeatCount?: number): Observable<T>;
		retry(retryCount?: number): Observable<T>;
		scan<TAcc>(seed: TAcc, accumulator: (acc: TAcc, value: T) => TAcc): Observable<TAcc>;
		scan(accumulator: (acc: T, value: T) => T): Observable<T>;
		skipLast(count: number): Observable<T>;
		startWith(...values: T[]): Observable<T>;
		startWith(scheduler: IScheduler, ...values: T[]): Observable<T>;
		takeLast(count: number): Observable<T>;
		takeLastBuffer(count: number): Observable<T[]>;

		select<TResult>(selector: (value: T, index: number, source: Observable<T>) => TResult, thisArg?: any): Observable<TResult>;
		map<TResult>(selector: (value: T, index: number, source: Observable<T>) => TResult, thisArg?: any): Observable<TResult>;	// alias for select
		pluck<TResult>(prop: string): Observable<TResult>;
		selectMany<TOther, TResult>(selector: (value: T) => Observable<TOther>, resultSelector: (item: T, other: TOther) => TResult): Observable<TResult>;
		selectMany<TOther, TResult>(selector: (value: T) => IPromise<TOther>, resultSelector: (item: T, other: TOther) => TResult): Observable<TResult>;
		selectMany<TResult>(selector: (value: T) => Observable<TResult>): Observable<TResult>;
		selectMany<TResult>(selector: (value: T) => IPromise<TResult>): Observable<TResult>;
		selectMany<TResult>(other: Observable<TResult>): Observable<TResult>;
		selectMany<TResult>(other: IPromise<TResult>): Observable<TResult>;
		flatMap<TOther, TResult>(selector: (value: T) => Observable<TOther>, resultSelector: (item: T, other: TOther) => TResult): Observable<TResult>;	// alias for selectMany
		flatMap<TOther, TResult>(selector: (value: T) => IPromise<TOther>, resultSelector: (item: T, other: TOther) => TResult): Observable<TResult>;	// alias for selectMany
		flatMap<TResult>(selector: (value: T) => Observable<TResult>): Observable<TResult>;	// alias for selectMany
		flatMap<TResult>(selector: (value: T) => IPromise<TResult>): Observable<TResult>;	// alias for selectMany
		flatMap<TResult>(other: Observable<TResult>): Observable<TResult>;	// alias for selectMany
		flatMap<TResult>(other: IPromise<TResult>): Observable<TResult>;	// alias for selectMany

		selectConcat<T2, R>(selector: (value: T, index: number) => Observable<T2>, resultSelector: (value1: T, value2: T2, index: number) => R): Observable<R>;
		selectConcat<T2, R>(selector: (value: T, index: number) => IPromise<T2>, resultSelector: (value1: T, value2: T2, index: number) => R): Observable<R>;
		selectConcat<R>(selector: (value: T, index: number) => Observable<R>): Observable<R>;
		selectConcat<R>(selector: (value: T, index: number) => IPromise<R>): Observable<R>;
		selectConcat<R>(sequence: Observable<R>): Observable<R>;

		/**
		*  Projects each element of an observable sequence into a new sequence of observable sequences by incorporating the element's index and then
		*  transforms an observable sequence of observable sequences into an observable sequence producing values only from the most recent observable sequence.
		* @param selector A transform function to apply to each source element; the second parameter of the function represents the index of the source element.
		* @param [thisArg] Object to use as this when executing callback.
		* @returns An observable sequence whose elements are the result of invoking the transform function on each element of source producing an Observable of Observable sequences
		*  and that at any point in time produces the elements of the most recent inner observable sequence that has been received.
		*/
		selectSwitch<TResult>(selector: (value: T, index: number, source: Observable<T>) => Observable<TResult>, thisArg?: any): Observable<TResult>;
		/**
		*  Projects each element of an observable sequence into a new sequence of observable sequences by incorporating the element's index and then
		*  transforms an observable sequence of observable sequences into an observable sequence producing values only from the most recent observable sequence.
		* @param selector A transform function to apply to each source element; the second parameter of the function represents the index of the source element.
		* @param [thisArg] Object to use as this when executing callback.
		* @returns An observable sequence whose elements are the result of invoking the transform function on each element of source producing an Observable of Observable sequences
		*  and that at any point in time produces the elements of the most recent inner observable sequence that has been received.
		*/
		flatMapLatest<TResult>(selector: (value: T, index: number, source: Observable<T>) => Observable<TResult>, thisArg?: any): Observable<TResult>;	// alias for selectSwitch
		/**
		*  Projects each element of an observable sequence into a new sequence of observable sequences by incorporating the element's index and then
		*  transforms an observable sequence of observable sequences into an observable sequence producing values only from the most recent observable sequence.
		* @param selector A transform function to apply to each source element; the second parameter of the function represents the index of the source element.
		* @param [thisArg] Object to use as this when executing callback.
		* @since 2.2.28
		* @returns An observable sequence whose elements are the result of invoking the transform function on each element of source producing an Observable of Observable sequences
		*  and that at any point in time produces the elements of the most recent inner observable sequence that has been received.
		*/
		switchMap<TResult>(selector: (value: T, index: number, source: Observable<T>) => TResult, thisArg?: any): Observable<TResult>;	// alias for selectSwitch

		skip(count: number): Observable<T>;
		skipWhile(predicate: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<T>;
		take(count: number, scheduler?: IScheduler): Observable<T>;
		takeWhile(predicate: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<T>;
		where(predicate: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<T>;
		filter(predicate: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<T>; // alias for where

		/**
		* Converts an existing observable sequence to an ES6 Compatible Promise
		* @example
		* var promise = Rx.Observable.return(42).toPromise(RSVP.Promise);
		* @param promiseCtor The constructor of the promise.
		* @returns An ES6 compatible promise with the last value from the observable sequence.
		*/
		toPromise<TPromise extends IPromise<T>>(promiseCtor: { new (resolver: (resolvePromise: (value: T) => void, rejectPromise: (reason: any) => void) => void): TPromise; }): TPromise;
		/**
		* Converts an existing observable sequence to an ES6 Compatible Promise
		* @example
		* var promise = Rx.Observable.return(42).toPromise(RSVP.Promise);
		*
		* // With config
		* Rx.config.Promise = RSVP.Promise;
		* var promise = Rx.Observable.return(42).toPromise();
		* @param [promiseCtor] The constructor of the promise. If not provided, it looks for it in Rx.config.Promise.
		* @returns An ES6 compatible promise with the last value from the observable sequence.
		*/
		toPromise(promiseCtor?: { new (resolver: (resolvePromise: (value: T) => void, rejectPromise: (reason: any) => void) => void): IPromise<T>; }): IPromise<T>;

		// Experimental Flattening

		/**
		* Performs a exclusive waiting for the first to finish before subscribing to another observable.
		* Observables that come in between subscriptions will be dropped on the floor.
		* Can be applied on `Observable<Observable<R>>` or `Observable<IPromise<R>>`.
		* @since 2.2.28
		* @returns A exclusive observable with only the results that happen when subscribed.
		*/
		exclusive<R>(): Observable<R>;

		/**
		* Performs a exclusive map waiting for the first to finish before subscribing to another observable.
		* Observables that come in between subscriptions will be dropped on the floor.
		* Can be applied on `Observable<Observable<I>>` or `Observable<IPromise<I>>`.
		* @since 2.2.28
		* @param selector Selector to invoke for every item in the current subscription.
		* @param [thisArg] An optional context to invoke with the selector parameter.
		* @returns {An exclusive observable with only the results that happen when subscribed.
		*/
		exclusiveMap<I, R>(selector: (value: I, index: number, source: Observable<I>) => R, thisArg?: any): Observable<R>;
	}

	interface ObservableStatic {
		create<T>(subscribe: (observer: Observer<T>) => IDisposable): Observable<T>;
		create<T>(subscribe: (observer: Observer<T>) => () => void): Observable<T>;
		create<T>(subscribe: (observer: Observer<T>) => void): Observable<T>;
		createWithDisposable<T>(subscribe: (observer: Observer<T>) => IDisposable): Observable<T>;
		defer<T>(observableFactory: () => Observable<T>): Observable<T>;
		defer<T>(observableFactory: () => IPromise<T>): Observable<T>;
		empty<T>(scheduler?: IScheduler): Observable<T>;

		/**
		* This method creates a new Observable sequence from an array object.
		* @param array An array-like or iterable object to convert to an Observable sequence.
		* @param mapFn Map function to call on every element of the array.
		* @param [thisArg] The context to use calling the mapFn if provided.
		* @param [scheduler] Optional scheduler to use for scheduling.  If not provided, defaults to Scheduler.currentThread.
		*/
		from<T, TResult>(array: T[], mapFn: (value: T, index: number) => TResult, thisArg?: any, scheduler?: IScheduler): Observable<TResult>;
		/**
		* This method creates a new Observable sequence from an array object.
		* @param array An array-like or iterable object to convert to an Observable sequence.
		* @param [mapFn] Map function to call on every element of the array.
		* @param [thisArg] The context to use calling the mapFn if provided.
		* @param [scheduler] Optional scheduler to use for scheduling.  If not provided, defaults to Scheduler.currentThread.
		*/
		from<T>(array: T[], mapFn?: (value: T, index: number) => T, thisArg?: any, scheduler?: IScheduler): Observable<T>;

		/**
		* This method creates a new Observable sequence from an array-like object.
		* @param array An array-like or iterable object to convert to an Observable sequence.
		* @param mapFn Map function to call on every element of the array.
		* @param [thisArg] The context to use calling the mapFn if provided.
		* @param [scheduler] Optional scheduler to use for scheduling.  If not provided, defaults to Scheduler.currentThread.
		*/
		from<T, TResult>(array: { length: number;[index: number]: T; }, mapFn: (value: T, index: number) => TResult, thisArg?: any, scheduler?: IScheduler): Observable<TResult>;
		/**
		* This method creates a new Observable sequence from an array-like object.
		* @param array An array-like or iterable object to convert to an Observable sequence.
		* @param [mapFn] Map function to call on every element of the array.
		* @param [thisArg] The context to use calling the mapFn if provided.
		* @param [scheduler] Optional scheduler to use for scheduling.  If not provided, defaults to Scheduler.currentThread.
		*/
		from<T>(array: { length: number;[index: number]: T; }, mapFn?: (value: T, index: number) => T, thisArg?: any, scheduler?: IScheduler): Observable<T>;

		/**
		* This method creates a new Observable sequence from an array-like or iterable object.
		* @param array An array-like or iterable object to convert to an Observable sequence.
		* @param [mapFn] Map function to call on every element of the array.
		* @param [thisArg] The context to use calling the mapFn if provided.
		* @param [scheduler] Optional scheduler to use for scheduling.  If not provided, defaults to Scheduler.currentThread.
		*/
		from<T>(iterable: any, mapFn?: (value: any, index: number) => T, thisArg?: any, scheduler?: IScheduler): Observable<T>;

		fromArray<T>(array: T[], scheduler?: IScheduler): Observable<T>;
		fromArray<T>(array: { length: number;[index: number]: T; }, scheduler?: IScheduler): Observable<T>;

		/**
		*  Converts an iterable into an Observable sequence
		*
		* @example
		*  var res = Rx.Observable.fromIterable(new Map());
		*  var res = Rx.Observable.fromIterable(function* () { yield 42; });
		*  var res = Rx.Observable.fromIterable(new Set(), Rx.Scheduler.timeout);
		* @param generator Generator to convert from.
		* @param [scheduler] Scheduler to run the enumeration of the input sequence on.
		* @returns The observable sequence whose elements are pulled from the given generator sequence.
		*/
		fromItreable<T>(generator: () => { next(): { done: boolean; value?: T; }; }, scheduler?: IScheduler): Observable<T>;

		/**
		*  Converts an iterable into an Observable sequence
		*
		* @example
		*  var res = Rx.Observable.fromIterable(new Map());
		*  var res = Rx.Observable.fromIterable(new Set(), Rx.Scheduler.timeout);
		* @param iterable Iterable to convert from.
		* @param [scheduler] Scheduler to run the enumeration of the input sequence on.
		* @returns The observable sequence whose elements are pulled from the given generator sequence.
		*/
		fromItreable<T>(iterable: {}, scheduler?: IScheduler): Observable<T>;	// todo: can't describe ES6 Iterable via TypeScript type system
		generate<TState, TResult>(initialState: TState, condition: (state: TState) => boolean, iterate: (state: TState) => TState, resultSelector: (state: TState) => TResult, scheduler?: IScheduler): Observable<TResult>;
		never<T>(): Observable<T>;

		/**
		*  This method creates a new Observable instance with a variable number of arguments, regardless of number or type of the arguments.
		*
		* @example
		*  var res = Rx.Observable.of(1, 2, 3);
		* @since 2.2.28
		* @returns The observable sequence whose elements are pulled from the given arguments.
		*/
		of<T>(...values: T[]): Observable<T>;

		/**
		*  This method creates a new Observable instance with a variable number of arguments, regardless of number or type of the arguments.
		* @example
		*  var res = Rx.Observable.ofWithScheduler(Rx.Scheduler.timeout, 1, 2, 3);
		* @since 2.2.28
		* @param [scheduler] A scheduler to use for scheduling the arguments.
		* @returns The observable sequence whose elements are pulled from the given arguments.
		*/
		ofWithScheduler<T>(scheduler?: IScheduler, ...values: T[]): Observable<T>;
		range(start: number, count: number, scheduler?: IScheduler): Observable<number>;
		repeat<T>(value: T, repeatCount?: number, scheduler?: IScheduler): Observable<T>;
		return<T>(value: T, scheduler?: IScheduler): Observable<T>;
		/**
		 * @since 2.2.28
		 */
		just<T>(value: T, scheduler?: IScheduler): Observable<T>;	// alias for return
		returnValue<T>(value: T, scheduler?: IScheduler): Observable<T>;	// alias for return
		throw<T>(exception: Error, scheduler?: IScheduler): Observable<T>;
		throw<T>(exception: any, scheduler?: IScheduler): Observable<T>;
		throwException<T>(exception: Error, scheduler?: IScheduler): Observable<T>;	// alias for throw
		throwException<T>(exception: any, scheduler?: IScheduler): Observable<T>;	// alias for throw
		throwError<T>(error: Error, scheduler?: IScheduler): Observable<T>;	// alias for throw
		throwError<T>(error: any, scheduler?: IScheduler): Observable<T>;	// alias for throw

		catch<T>(sources: Observable<T>[]): Observable<T>;
		catch<T>(sources: IPromise<T>[]): Observable<T>;
		catchException<T>(sources: Observable<T>[]): Observable<T>;	// alias for catch
		catchException<T>(sources: IPromise<T>[]): Observable<T>;	// alias for catch
		catchError<T>(sources: Observable<T>[]): Observable<T>;	// alias for catch
		catchError<T>(sources: IPromise<T>[]): Observable<T>;	// alias for catch
		catch<T>(...sources: Observable<T>[]): Observable<T>;
		catch<T>(...sources: IPromise<T>[]): Observable<T>;
		catchException<T>(...sources: Observable<T>[]): Observable<T>;	// alias for catch
		catchException<T>(...sources: IPromise<T>[]): Observable<T>;	// alias for catch
		catchError<T>(...sources: Observable<T>[]): Observable<T>;	// alias for catch
		catchError<T>(...sources: IPromise<T>[]): Observable<T>;	// alias for catch

		combineLatest<T, T2, TResult>(first: Observable<T>, second: Observable<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		combineLatest<T, T2, TResult>(first: IPromise<T>, second: Observable<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		combineLatest<T, T2, TResult>(first: Observable<T>, second: IPromise<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		combineLatest<T, T2, TResult>(first: IPromise<T>, second: IPromise<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, TResult>(first: Observable<T>, second: Observable<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, TResult>(first: Observable<T>, second: Observable<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, TResult>(first: Observable<T>, second: IPromise<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, TResult>(first: Observable<T>, second: IPromise<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, TResult>(first: IPromise<T>, second: Observable<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, TResult>(first: IPromise<T>, second: Observable<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, TResult>(first: IPromise<T>, second: IPromise<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, TResult>(first: IPromise<T>, second: IPromise<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: Observable<T>, second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: Observable<T>, second: Observable<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: Observable<T>, second: Observable<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: Observable<T>, second: Observable<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: Observable<T>, second: IPromise<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: Observable<T>, second: IPromise<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: Observable<T>, second: IPromise<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: Observable<T>, second: IPromise<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: IPromise<T>, second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: IPromise<T>, second: Observable<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: IPromise<T>, second: Observable<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: IPromise<T>, second: Observable<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: IPromise<T>, second: IPromise<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: IPromise<T>, second: IPromise<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: IPromise<T>, second: IPromise<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, TResult>(first: IPromise<T>, second: IPromise<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		combineLatest<T, T2, T3, T4, T5, TResult>(first: Observable<T>, second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, fifth: Observable<T5>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4, v5: T5) => TResult): Observable<TResult>;
		combineLatest<TOther, TResult>(souces: Observable<TOther>[], resultSelector: (...otherValues: TOther[]) => TResult): Observable<TResult>;
		combineLatest<TOther, TResult>(souces: IPromise<TOther>[], resultSelector: (...otherValues: TOther[]) => TResult): Observable<TResult>;

		withLatestFrom<T, T2, TResult>(first: Observable<T>, second: Observable<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, TResult>(first: IPromise<T>, second: Observable<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, TResult>(first: Observable<T>, second: IPromise<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, TResult>(first: IPromise<T>, second: IPromise<T2>, resultSelector: (v1: T, v2: T2) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, TResult>(first: Observable<T>, second: Observable<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, TResult>(first: Observable<T>, second: Observable<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, TResult>(first: Observable<T>, second: IPromise<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, TResult>(first: Observable<T>, second: IPromise<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, TResult>(first: IPromise<T>, second: Observable<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, TResult>(first: IPromise<T>, second: Observable<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, TResult>(first: IPromise<T>, second: IPromise<T2>, third: Observable<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, TResult>(first: IPromise<T>, second: IPromise<T2>, third: IPromise<T3>, resultSelector: (v1: T, v2: T2, v3: T3) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: Observable<T>, second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: Observable<T>, second: Observable<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: Observable<T>, second: Observable<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: Observable<T>, second: Observable<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: Observable<T>, second: IPromise<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: Observable<T>, second: IPromise<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: Observable<T>, second: IPromise<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: Observable<T>, second: IPromise<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: IPromise<T>, second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: IPromise<T>, second: Observable<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: IPromise<T>, second: Observable<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: IPromise<T>, second: Observable<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: IPromise<T>, second: IPromise<T2>, third: Observable<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: IPromise<T>, second: IPromise<T2>, third: Observable<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: IPromise<T>, second: IPromise<T2>, third: IPromise<T3>, fourth: Observable<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, TResult>(first: IPromise<T>, second: IPromise<T2>, third: IPromise<T3>, fourth: IPromise<T4>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4) => TResult): Observable<TResult>;
		withLatestFrom<T, T2, T3, T4, T5, TResult>(first: Observable<T>, second: Observable<T2>, third: Observable<T3>, fourth: Observable<T4>, fifth: Observable<T5>, resultSelector: (v1: T, v2: T2, v3: T3, v4: T4, v5: T5) => TResult): Observable<TResult>;
		withLatestFrom<TOther, TResult>(souces: Observable<TOther>[], resultSelector: (...otherValues: TOther[]) => TResult): Observable<TResult>;
		withLatestFrom<TOther, TResult>(souces: IPromise<TOther>[], resultSelector: (...otherValues: TOther[]) => TResult): Observable<TResult>;

		concat<T>(...sources: Observable<T>[]): Observable<T>;
		concat<T>(...sources: IPromise<T>[]): Observable<T>;
		concat<T>(sources: Observable<T>[]): Observable<T>;
		concat<T>(sources: IPromise<T>[]): Observable<T>;
		merge<T>(...sources: Observable<T>[]): Observable<T>;
		merge<T>(...sources: IPromise<T>[]): Observable<T>;
		merge<T>(sources: Observable<T>[]): Observable<T>;
		merge<T>(sources: IPromise<T>[]): Observable<T>;
		merge<T>(scheduler: IScheduler, ...sources: Observable<T>[]): Observable<T>;
		merge<T>(scheduler: IScheduler, ...sources: IPromise<T>[]): Observable<T>;
		merge<T>(scheduler: IScheduler, sources: Observable<T>[]): Observable<T>;
		merge<T>(scheduler: IScheduler, sources: IPromise<T>[]): Observable<T>;

		zip<T1, T2, TResult>(first: Observable<T1>, sources: Observable<T2>[], resultSelector: (item1: T1, ...right: T2[]) => TResult): Observable<TResult>;
		zip<T1, T2, TResult>(first: Observable<T1>, sources: IPromise<T2>[], resultSelector: (item1: T1, ...right: T2[]) => TResult): Observable<TResult>;
		zip<T1, T2, TResult>(source1: Observable<T1>, source2: Observable<T2>, resultSelector: (item1: T1, item2: T2) => TResult): Observable<TResult>;
		zip<T1, T2, TResult>(source1: Observable<T1>, source2: IPromise<T2>, resultSelector: (item1: T1, item2: T2) => TResult): Observable<TResult>;
		zip<T1, T2, T3, TResult>(source1: Observable<T1>, source2: Observable<T2>, source3: Observable<T3>, resultSelector: (item1: T1, item2: T2, item3: T3) => TResult): Observable<TResult>;
		zip<T1, T2, T3, TResult>(source1: Observable<T1>, source2: Observable<T2>, source3: IPromise<T3>, resultSelector: (item1: T1, item2: T2, item3: T3) => TResult): Observable<TResult>;
		zip<T1, T2, T3, TResult>(source1: Observable<T1>, source2: IPromise<T2>, source3: Observable<T3>, resultSelector: (item1: T1, item2: T2, item3: T3) => TResult): Observable<TResult>;
		zip<T1, T2, T3, TResult>(source1: Observable<T1>, source2: IPromise<T2>, source3: IPromise<T3>, resultSelector: (item1: T1, item2: T2, item3: T3) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, TResult>(source1: Observable<T1>, source2: Observable<T2>, source3: Observable<T3>, source4: Observable<T4>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, TResult>(source1: Observable<T1>, source2: Observable<T2>, source3: Observable<T3>, source4: IPromise<T4>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, TResult>(source1: Observable<T1>, source2: Observable<T2>, source3: IPromise<T3>, source4: Observable<T4>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, TResult>(source1: Observable<T1>, source2: Observable<T2>, source3: IPromise<T3>, source4: IPromise<T4>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, TResult>(source1: Observable<T1>, source2: IPromise<T2>, source3: Observable<T3>, source4: Observable<T4>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, TResult>(source1: Observable<T1>, source2: IPromise<T2>, source3: Observable<T3>, source4: IPromise<T4>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, TResult>(source1: Observable<T1>, source2: IPromise<T2>, source3: IPromise<T3>, source4: Observable<T4>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, TResult>(source1: Observable<T1>, source2: IPromise<T2>, source3: IPromise<T3>, source4: IPromise<T4>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4) => TResult): Observable<TResult>;
		zip<T1, T2, T3, T4, T5, TResult>(source1: Observable<T1>, source2: Observable<T2>, source3: Observable<T3>, source4: Observable<T4>, source5: Observable<T5>, resultSelector: (item1: T1, item2: T2, item3: T3, item4: T4, item5: T5) => TResult): Observable<TResult>;
		zipArray<T>(...sources: Observable<T>[]): Observable<T[]>;
		zipArray<T>(sources: Observable<T>[]): Observable<T[]>;

		/**
		* Converts a Promise to an Observable sequence
		* @param promise An ES6 Compliant promise.
		* @returns An Observable sequence which wraps the existing promise success and failure.
		*/
		fromPromise<T>(promise: IPromise<T>): Observable<T>;
	}

	export var Observable: ObservableStatic;

	interface ISubject<T> extends Observable<T>, Observer<T>, IDisposable {
		hasObservers(): boolean;
	}

    export interface Subject<T> extends ISubject<T> {
    }

    interface SubjectStatic {
        new <T>(): Subject<T>;
		create<T>(observer?: Observer<T>, observable?: Observable<T>): ISubject<T>;
	}

	export var Subject: SubjectStatic;

	export interface AsyncSubject<T> extends Subject<T> {
	}

	interface AsyncSubjectStatic {
		new <T>(): AsyncSubject<T>;
	}

	export var AsyncSubject: AsyncSubjectStatic;
}
