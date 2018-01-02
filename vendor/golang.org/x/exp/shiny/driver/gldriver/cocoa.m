// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build 386 amd64
// +build !ios

#include "_cgo_export.h"
#include <pthread.h>
#include <stdio.h>

#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <OpenGL/gl3.h>

void makeCurrentContext(uintptr_t context) {
	NSOpenGLContext* ctx = (NSOpenGLContext*)context;
	[ctx makeCurrentContext];
}

void flushContext(uintptr_t context) {
	NSOpenGLContext* ctx = (NSOpenGLContext*)context;
	[ctx flushBuffer];
}

uint64 threadID() {
	uint64 id;
	if (pthread_threadid_np(pthread_self(), &id)) {
		abort();
	}
	return id;
}

@interface ScreenGLView : NSOpenGLView<NSWindowDelegate>
{
}
@end

@implementation ScreenGLView
- (void)prepareOpenGL {
	[self setWantsBestResolutionOpenGLSurface:YES];
	GLint swapInt = 1;
	NSOpenGLContext *ctx = [self openGLContext];
	[ctx setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];

	// Using attribute arrays in OpenGL 3.3 requires the use of a VBA.
	// But VBAs don't exist in ES 2. So we bind a default one.
	GLuint vba;
	glGenVertexArrays(1, &vba);
	glBindVertexArray(vba);

	preparedOpenGL((GoUintptr)self, (GoUintptr)ctx, (GoUintptr)vba);
}

- (void)callSetGeom {
	// Calculate screen PPI.
	//
	// Note that the backingScaleFactor converts from logical
	// pixels to actual pixels, but both of these units vary
	// independently from real world size. E.g.
	//
	// 13" Retina Macbook Pro, 2560x1600, 227ppi, backingScaleFactor=2, scale=3.15
	// 15" Retina Macbook Pro, 2880x1800, 220ppi, backingScaleFactor=2, scale=3.06
	// 27" iMac,               2560x1440, 109ppi, backingScaleFactor=1, scale=1.51
	// 27" Retina iMac,        5120x2880, 218ppi, backingScaleFactor=2, scale=3.03
	NSScreen *screen = self.window.screen;
	double screenPixW = [screen frame].size.width * [screen backingScaleFactor];

	CGDirectDisplayID display = (CGDirectDisplayID)[[[screen deviceDescription] valueForKey:@"NSScreenNumber"] intValue];
	CGSize screenSizeMM = CGDisplayScreenSize(display); // in millimeters
	float ppi = 25.4 * screenPixW / screenSizeMM.width;
	float pixelsPerPt = ppi/72.0;

	// The width and height reported to the geom package are the
	// bounds of the OpenGL view. Several steps are necessary.
	// First, [self bounds] gives us the number of logical pixels
	// in the view. Multiplying this by the backingScaleFactor
	// gives us the number of actual pixels.
	NSRect r = [self bounds];
	int w = r.size.width * [screen backingScaleFactor];
	int h = r.size.height * [screen backingScaleFactor];

	setGeom((GoUintptr)self, pixelsPerPt, w, h);
}

- (void)reshape {
	[super reshape];
	[self callSetGeom];
}

- (void)drawRect:(NSRect)theRect {
	// Called during resize. Do an extra draw if we are visible.
	// This gets rid of flicker when resizing.
	drawgl((GoUintptr)self);
}

- (void)mouseEventNS:(NSEvent *)theEvent {
	NSPoint p = [theEvent locationInWindow];
	double h = self.frame.size.height;

	// Both h and p are measured in Cocoa pixels, which are a fraction of
	// physical pixels, so we multiply by backingScaleFactor.
	double scale = [self.window.screen backingScaleFactor];

	double x = p.x * scale;
	double y = (h - p.y) * scale - 1; // flip origin from bottom-left to top-left.

	double dx, dy;
	if (theEvent.type == NSScrollWheel) {
		dx = theEvent.scrollingDeltaX;
		dy = theEvent.scrollingDeltaY;
	}

	mouseEvent((GoUintptr)self, x, y, dx, dy, theEvent.type, theEvent.buttonNumber, theEvent.modifierFlags);
}

- (void)mouseMoved:(NSEvent *)theEvent        { [self mouseEventNS:theEvent]; }
- (void)mouseDown:(NSEvent *)theEvent         { [self mouseEventNS:theEvent]; }
- (void)mouseUp:(NSEvent *)theEvent           { [self mouseEventNS:theEvent]; }
- (void)mouseDragged:(NSEvent *)theEvent      { [self mouseEventNS:theEvent]; }
- (void)rightMouseDown:(NSEvent *)theEvent    { [self mouseEventNS:theEvent]; }
- (void)rightMouseUp:(NSEvent *)theEvent      { [self mouseEventNS:theEvent]; }
- (void)rightMouseDragged:(NSEvent *)theEvent { [self mouseEventNS:theEvent]; }
- (void)otherMouseDown:(NSEvent *)theEvent    { [self mouseEventNS:theEvent]; }
- (void)otherMouseUp:(NSEvent *)theEvent      { [self mouseEventNS:theEvent]; }
- (void)otherMouseDragged:(NSEvent *)theEvent { [self mouseEventNS:theEvent]; }
- (void)scrollWheel:(NSEvent *)theEvent       { [self mouseEventNS:theEvent]; }

// raw modifier key presses
- (void)flagsChanged:(NSEvent *)theEvent {
	flagEvent((GoUintptr)self, theEvent.modifierFlags);
}

// overrides special handling of escape and tab
- (BOOL)performKeyEquivalent:(NSEvent *)theEvent {
	[self key:theEvent];
	return YES;
}

- (void)keyDown:(NSEvent *)theEvent { [self key:theEvent]; }
- (void)keyUp:(NSEvent *)theEvent   { [self key:theEvent]; }

- (void)key:(NSEvent *)theEvent {
	NSRange range = [theEvent.characters rangeOfComposedCharacterSequenceAtIndex:0];

	uint8_t buf[4] = {0, 0, 0, 0};
	if (![theEvent.characters getBytes:buf
			maxLength:4
			usedLength:nil
			encoding:NSUTF32LittleEndianStringEncoding
			options:NSStringEncodingConversionAllowLossy
			range:range
			remainingRange:nil]) {
		NSLog(@"failed to read key event %@", theEvent);
		return;
	}

	uint32_t rune = (uint32_t)buf[0]<<0 | (uint32_t)buf[1]<<8 | (uint32_t)buf[2]<<16 | (uint32_t)buf[3]<<24;

	uint8_t direction;
	if ([theEvent isARepeat]) {
		direction = 0;
	} else if (theEvent.type == NSKeyDown) {
		direction = 1;
	} else {
		direction = 2;
	}
	keyEvent((GoUintptr)self, (int32_t)rune, direction, theEvent.keyCode, theEvent.modifierFlags);
}

- (void)windowDidChangeScreenProfile:(NSNotification *)notification {
	[self callSetGeom];
}

- (void)windowDidExpose:(NSNotification *)notification {
	lifecycleVisible((GoUintptr)self);
}

- (void)windowDidBecomeKey:(NSNotification *)notification {
	lifecycleFocused((GoUintptr)self);
}

- (void)windowDidResignKey:(NSNotification *)notification {
	if (![NSApp isHidden]) {
		lifecycleVisible((GoUintptr)self);
	}
}

- (void)windowWillClose:(NSNotification *)notification {
	if (self.window.nextResponder == NULL) {
		return; // already called close
	}
	windowClosing((GoUintptr)self);
	[self.window.nextResponder release];
	self.window.nextResponder = NULL;
}
@end

@interface AppDelegate : NSObject<NSApplicationDelegate>
{
}
@end

@implementation AppDelegate
- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
	driverStarted();
	[[NSRunningApplication currentApplication] activateWithOptions:(NSApplicationActivateAllWindows | NSApplicationActivateIgnoringOtherApps)];
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
	lifecycleDeadAll();
}

- (void)applicationWillHide:(NSNotification *)aNotification {
	lifecycleAliveAll();
}
@end

uintptr_t doNewWindow(int width, int height) {
	NSScreen *screen = [NSScreen mainScreen];
	double w = (double)width / [screen backingScaleFactor];
	double h = (double)height / [screen backingScaleFactor];
	__block ScreenGLView* view = NULL;

	dispatch_sync(dispatch_get_main_queue(), ^{
		id menuBar = [NSMenu new];
		id menuItem = [NSMenuItem new];
		[menuBar addItem:menuItem];
		[NSApp setMainMenu:menuBar];

		id menu = [NSMenu new];
		NSString* name = [[NSProcessInfo processInfo] processName];

		id hideMenuItem = [[NSMenuItem alloc] initWithTitle:@"Hide"
			action:@selector(hide:) keyEquivalent:@"h"];
		[menu addItem:hideMenuItem];

		id quitMenuItem = [[NSMenuItem alloc] initWithTitle:@"Quit"
			action:@selector(terminate:) keyEquivalent:@"q"];
		[menu addItem:quitMenuItem];
		[menuItem setSubmenu:menu];

		NSRect rect = NSMakeRect(0, 0, w, h);

		NSWindow* window = [[NSWindow alloc] initWithContentRect:rect
				styleMask:NSTitledWindowMask
				backing:NSBackingStoreBuffered
				defer:NO];
		window.styleMask |= NSResizableWindowMask;
		window.styleMask |= NSMiniaturizableWindowMask ;
		window.styleMask |= NSClosableWindowMask;
		window.title = name;
		window.displaysWhenScreenProfileChanges = YES;
		[window cascadeTopLeftFromPoint:NSMakePoint(20,20)];
		[window setAcceptsMouseMovedEvents:YES];

		NSOpenGLPixelFormatAttribute attr[] = {
			NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
			NSOpenGLPFAColorSize,     24,
			NSOpenGLPFAAlphaSize,     8,
			NSOpenGLPFADepthSize,     16,
			NSOpenGLPFAAccelerated,
			NSOpenGLPFADoubleBuffer,
			NSOpenGLPFAAllowOfflineRenderers,
			0
		};
		id pixFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attr];
		view = [[ScreenGLView alloc] initWithFrame:rect pixelFormat:pixFormat];
		[window setContentView:view];
		[window setDelegate:view];
		[window makeFirstResponder:view];
	});

	return (uintptr_t)view;
}

void doShowWindow(uintptr_t viewID) {
	ScreenGLView* view = (ScreenGLView*)viewID;
	dispatch_async(dispatch_get_main_queue(), ^{
		[view.window makeKeyAndOrderFront:view.window];
	});
}

void doCloseWindow(uintptr_t viewID) {
	ScreenGLView* view = (ScreenGLView*)viewID;
	dispatch_sync(dispatch_get_main_queue(), ^{
		[view.window performClose:view];
	});
}

void startDriver() {
	[NSAutoreleasePool new];
	[NSApplication sharedApplication];
	[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
	AppDelegate* delegate = [[AppDelegate alloc] init];
	[NSApp setDelegate:delegate];
	[NSApp run];
}

void stopDriver() {
	dispatch_async(dispatch_get_main_queue(), ^{
		[NSApp terminate:nil];
	});
}
