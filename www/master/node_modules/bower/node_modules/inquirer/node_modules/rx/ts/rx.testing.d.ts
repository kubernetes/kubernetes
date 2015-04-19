// Type definitions for RxJS-Testing v2.2.28
// Project: https://github.com/Reactive-Extensions/RxJS/
// Definitions by: Igor Oleinikov <https://github.com/Igorbek>
// Definitions: https://github.com/borisyankov/DefinitelyTyped

///<reference path="rx.d.ts" />
///<reference path="rx.virtualtime.d.ts" />

declare module Rx {
	export interface TestScheduler extends VirtualTimeScheduler<number, number> {
		createColdObservable<T>(...records: Recorded[]): Observable<T>;
		createHotObservable<T>(...records: Recorded[]): Observable<T>;
		createObserver<T>(): MockObserver<T>;

		startWithTiming<T>(create: () => Observable<T>, createdAt: number, subscribedAt: number, disposedAt: number): MockObserver<T>;
		startWithDispose<T>(create: () => Observable<T>, disposedAt: number): MockObserver<T>;
		startWithCreate<T>(create: () => Observable<T>): MockObserver<T>;
	}

	export var TestScheduler: {
		new (): TestScheduler;
	};

	export class Recorded {
		constructor(time: number, value: any, equalityComparer?: (x: any, y: any) => boolean);
		equals(other: Recorded): boolean;
		toString(): string;
		time: number;
		value: any;
	}

	export var ReactiveTest: {
		created: number;
		subscribed: number;
		disposed: number;

		onNext(ticks: number, value: any): Recorded;
		onNext(ticks: number, predicate: (value: any) => boolean): Recorded;
		onError(ticks: number, exception: any): Recorded;
		onError(ticks: number, predicate: (exception: any) => boolean): Recorded;
		onCompleted(ticks: number): Recorded;

		subscribe(subscribeAt: number, unsubscribeAt?: number): Subscription;
	};

	export class Subscription {
		constructor(subscribeAt: number, unsubscribeAt?: number);
		equals(other: Subscription): boolean;
	}

	export interface MockObserver<T> extends Observer<T> {
		messages: Recorded[];
	}

	interface MockObserverStatic extends ObserverStatic {
		new <T>(scheduler: IScheduler): MockObserver<T>;
	}

	export var MockObserver: MockObserverStatic;
}

declare module "rx.testing" {
	export = Rx;
}
