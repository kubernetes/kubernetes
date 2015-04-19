// Type definitions for RxJS-Async v2.2.28
// Project: http://rx.codeplex.com/
// Definitions by: zoetrope <https://github.com/zoetrope>, Igor Oleinikov <https://github.com/Igorbek>
// Definitions: https://github.com/borisyankov/DefinitelyTyped

///<reference path="rx.d.ts" />
///<reference path="rx.async-lite.d.ts" />

declare module Rx {
    interface ObservableStatic {
		start<T>(func: () => T, context?: any, scheduler?: IScheduler): Observable<T>;

        toAsync<TResult>(func: () => TResult, context?: any, scheduler?: IScheduler): () => Observable<TResult>;
		toAsync<T1, TResult>(func: (arg1: T1) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1) => Observable<TResult>;
		toAsync<T1, TResult>(func: (arg1?: T1) => TResult, context?: any, scheduler?: IScheduler): (arg1?: T1) => Observable<TResult>;
		toAsync<T1, TResult>(func: (...args: T1[]) => TResult, context?: any, scheduler?: IScheduler): (...args: T1[]) => Observable<TResult>;
		toAsync<T1, T2, TResult>(func: (arg1: T1, arg2: T2) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2) => Observable<TResult>;
		toAsync<T1, T2, TResult>(func: (arg1: T1, arg2?: T2) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2?: T2) => Observable<TResult>;
		toAsync<T1, T2, TResult>(func: (arg1?: T1, arg2?: T2) => TResult, context?: any, scheduler?: IScheduler): (arg1?: T1, arg2?: T2) => Observable<TResult>;
		toAsync<T1, T2, TResult>(func: (arg1: T1, ...args: T2[]) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, ...args: T2[]) => Observable<TResult>;
		toAsync<T1, T2, TResult>(func: (arg1?: T1, ...args: T2[]) => TResult, context?: any, scheduler?: IScheduler): (arg1?: T1, ...args: T2[]) => Observable<TResult>;
		toAsync<T1, T2, T3, TResult>(func: (arg1: T1, arg2: T2, arg3: T3) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2, arg3: T3) => Observable<TResult>;
		toAsync<T1, T2, T3, TResult>(func: (arg1: T1, arg2: T2, arg3?: T3) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2, arg3?: T3) => Observable<TResult>;
		toAsync<T1, T2, T3, TResult>(func: (arg1: T1, arg2?: T2, arg3?: T3) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2?: T2, arg3?: T3) => Observable<TResult>;
		toAsync<T1, T2, T3, TResult>(func: (arg1?: T1, arg2?: T2, arg3?: T3) => TResult, context?: any, scheduler?: IScheduler): (arg1?: T1, arg2?: T2, arg3?: T3) => Observable<TResult>;
		toAsync<T1, T2, T3, TResult>(func: (arg1: T1, arg2: T2, ...args: T3[]) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2, ...args: T3[]) => Observable<TResult>;
		toAsync<T1, T2, T3, TResult>(func: (arg1: T1, arg2?: T2, ...args: T3[]) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2?: T2, ...args: T3[]) => Observable<TResult>;
		toAsync<T1, T2, T3, TResult>(func: (arg1?: T1, arg2?: T2, ...args: T3[]) => TResult, context?: any, scheduler?: IScheduler): (arg1?: T1, arg2?: T2, ...args: T3[]) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1: T1, arg2: T2, arg3: T3, arg4: T4) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2, arg3: T3, arg4: T4) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1: T1, arg2: T2, arg3: T3, arg4?: T4) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2, arg3: T3, arg4?: T4) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1: T1, arg2: T2, arg3?: T3, arg4?: T4) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2, arg3?: T3, arg4?: T4) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1: T1, arg2?: T2, arg3?: T3, arg4?: T4) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2?: T2, arg3?: T3, arg4?: T4) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1?: T1, arg2?: T2, arg3?: T3, arg4?: T4) => TResult, context?: any, scheduler?: IScheduler): (arg1?: T1, arg2?: T2, arg3?: T3, arg4?: T4) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1: T1, arg2: T2, arg3: T3, ...args: T4[]) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2, arg3: T3, ...args: T4[]) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1: T1, arg2: T2, arg3?: T3, ...args: T4[]) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2: T2, arg3?: T3, ...args: T4[]) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1: T1, arg2?: T2, arg3?: T3, ...args: T4[]) => TResult, context?: any, scheduler?: IScheduler): (arg1: T1, arg2?: T2, arg3?: T3, ...args: T4[]) => Observable<TResult>;
		toAsync<T1, T2, T3, T4, TResult>(func: (arg1?: T1, arg2?: T2, arg3?: T3, ...args: T4[]) => TResult, context?: any, scheduler?: IScheduler): (arg1?: T1, arg2?: T2, arg3?: T3, ...args: T4[]) => Observable<TResult>;
	}
}

declare module "rx.async" {
	export = Rx;
}
