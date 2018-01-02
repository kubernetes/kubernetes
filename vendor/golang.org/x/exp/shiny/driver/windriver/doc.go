// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package windriver provides the Windows driver for accessing a screen.
package windriver // import "golang.org/x/exp/shiny/driver/windriver"

/*
Implementation Details

On Windows, GUI is managed via user code and OS sending messages to
a window. These messages include paint events, input events and others.
Any thread that hosts GUI must handle incoming window messages through
a "message loop".

windriver designates the thread that calls Main as the GUI thread.
It locks this thread, creates a special window to handle screen.Screen
calls and runs message loop. All new windows are created by the
same thread, so message loop above handles all their window messages.

Some general Windows rules about thread affinity of GUI objects:

part 1: Window handles
https://blogs.msdn.microsoft.com/oldnewthing/20051010-09/?p=33843

part 2: Device contexts
https://blogs.msdn.microsoft.com/oldnewthing/20051011-10/?p=33823

part 3: Menus, icons, cursors, and accelerator tables
https://blogs.msdn.microsoft.com/oldnewthing/20051012-00/?p=33803

part 4: GDI objects and other notes on affinity
https://blogs.msdn.microsoft.com/oldnewthing/20051013-11/?p=33783

part 5: Object clean-up
https://blogs.msdn.microsoft.com/oldnewthing/20051014-19/?p=33763

How to build Windows GUI articles:

http://www.codeproject.com/Articles/1988/Guide-to-WIN-Paint-for-Beginners
http://www.codeproject.com/Articles/2078/Guide-to-WIN-Paint-for-Intermediates
http://www.codeproject.com/Articles/224754/Guide-to-Win-Memory-DC

*/
