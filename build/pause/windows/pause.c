/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <windows.h>
#include <stdio.h>
#include <string.h>

#define STRINGIFY(x) #x
#define VERSION_STRING(x) STRINGIFY(x)

#ifndef VERSION
#define VERSION HEAD
#endif

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
	switch (fdwCtrlType)
	{
	case CTRL_C_EVENT:
		fprintf(stderr, "Shutting down, got signal\n");
		exit(0);

	case CTRL_BREAK_EVENT:
		fprintf(stderr, "Shutting down, got signal\n");
		exit(0);

	default:
		return FALSE;
	}
}

int main(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; ++i)
	{
		if (!_stricmp(argv[i], "-v"))
		{
			fprintf(stdout, "pause.c %s\n", VERSION_STRING(VERSION));
			return 0;
		}
	}

	if (SetConsoleCtrlHandler(CtrlHandler, TRUE))
	{
		Sleep(INFINITE);
	}
	else
	{
		printf("\nERROR: Could not set control handler\n");
		return 1;
	}
	return 0;
}
