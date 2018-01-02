// +build windows

package backuptar

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/go-winio/archive/tar" // until archive/tar supports pax extensions in its interface
)

const (
	c_ISUID  = 04000   // Set uid
	c_ISGID  = 02000   // Set gid
	c_ISVTX  = 01000   // Save text (sticky bit)
	c_ISDIR  = 040000  // Directory
	c_ISFIFO = 010000  // FIFO
	c_ISREG  = 0100000 // Regular file
	c_ISLNK  = 0120000 // Symbolic link
	c_ISBLK  = 060000  // Block special file
	c_ISCHR  = 020000  // Character special file
	c_ISSOCK = 0140000 // Socket
)

const (
	hdrFileAttributes        = "fileattr"
	hdrSecurityDescriptor    = "sd"
	hdrRawSecurityDescriptor = "rawsd"
	hdrMountPoint            = "mountpoint"
	hdrEaPrefix              = "xattr."
)

func writeZeroes(w io.Writer, count int64) error {
	buf := make([]byte, 8192)
	c := len(buf)
	for i := int64(0); i < count; i += int64(c) {
		if int64(c) > count-i {
			c = int(count - i)
		}
		_, err := w.Write(buf[:c])
		if err != nil {
			return err
		}
	}
	return nil
}

func copySparse(t *tar.Writer, br *winio.BackupStreamReader) error {
	curOffset := int64(0)
	for {
		bhdr, err := br.Next()
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		if err != nil {
			return err
		}
		if bhdr.Id != winio.BackupSparseBlock {
			return fmt.Errorf("unexpected stream %d", bhdr.Id)
		}

		// archive/tar does not support writing sparse files
		// so just write zeroes to catch up to the current offset.
		err = writeZeroes(t, bhdr.Offset-curOffset)
		if bhdr.Size == 0 {
			break
		}
		n, err := io.Copy(t, br)
		if err != nil {
			return err
		}
		curOffset = bhdr.Offset + n
	}
	return nil
}

// BasicInfoHeader creates a tar header from basic file information.
func BasicInfoHeader(name string, size int64, fileInfo *winio.FileBasicInfo) *tar.Header {
	hdr := &tar.Header{
		Name:         filepath.ToSlash(name),
		Size:         size,
		Typeflag:     tar.TypeReg,
		ModTime:      time.Unix(0, fileInfo.LastWriteTime.Nanoseconds()),
		ChangeTime:   time.Unix(0, fileInfo.ChangeTime.Nanoseconds()),
		AccessTime:   time.Unix(0, fileInfo.LastAccessTime.Nanoseconds()),
		CreationTime: time.Unix(0, fileInfo.CreationTime.Nanoseconds()),
		Winheaders:   make(map[string]string),
	}
	hdr.Winheaders[hdrFileAttributes] = fmt.Sprintf("%d", fileInfo.FileAttributes)

	if (fileInfo.FileAttributes & syscall.FILE_ATTRIBUTE_DIRECTORY) != 0 {
		hdr.Mode |= c_ISDIR
		hdr.Size = 0
		hdr.Typeflag = tar.TypeDir
	}
	return hdr
}

// WriteTarFileFromBackupStream writes a file to a tar writer using data from a Win32 backup stream.
//
// This encodes Win32 metadata as tar pax vendor extensions starting with MSWINDOWS.
//
// The additional Win32 metadata is:
//
// MSWINDOWS.fileattr: The Win32 file attributes, as a decimal value
//
// MSWINDOWS.rawsd: The Win32 security descriptor, in raw binary format
//
// MSWINDOWS.mountpoint: If present, this is a mount point and not a symlink, even though the type is '2' (symlink)
func WriteTarFileFromBackupStream(t *tar.Writer, r io.Reader, name string, size int64, fileInfo *winio.FileBasicInfo) error {
	name = filepath.ToSlash(name)
	hdr := BasicInfoHeader(name, size, fileInfo)

	// If r can be seeked, then this function is two-pass: pass 1 collects the
	// tar header data, and pass 2 copies the data stream. If r cannot be
	// seeked, then some header data (in particular EAs) will be silently lost.
	var (
		restartPos int64
		err        error
	)
	sr, readTwice := r.(io.Seeker)
	if readTwice {
		if restartPos, err = sr.Seek(0, io.SeekCurrent); err != nil {
			readTwice = false
		}
	}

	br := winio.NewBackupStreamReader(r)
	var dataHdr *winio.BackupHeader
	for dataHdr == nil {
		bhdr, err := br.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		switch bhdr.Id {
		case winio.BackupData:
			hdr.Mode |= c_ISREG
			if !readTwice {
				dataHdr = bhdr
			}
		case winio.BackupSecurity:
			sd, err := ioutil.ReadAll(br)
			if err != nil {
				return err
			}
			hdr.Winheaders[hdrRawSecurityDescriptor] = base64.StdEncoding.EncodeToString(sd)

		case winio.BackupReparseData:
			hdr.Mode |= c_ISLNK
			hdr.Typeflag = tar.TypeSymlink
			reparseBuffer, err := ioutil.ReadAll(br)
			rp, err := winio.DecodeReparsePoint(reparseBuffer)
			if err != nil {
				return err
			}
			if rp.IsMountPoint {
				hdr.Winheaders[hdrMountPoint] = "1"
			}
			hdr.Linkname = rp.Target

		case winio.BackupEaData:
			eab, err := ioutil.ReadAll(br)
			if err != nil {
				return err
			}
			eas, err := winio.DecodeExtendedAttributes(eab)
			if err != nil {
				return err
			}
			for _, ea := range eas {
				// Use base64 encoding for the binary value. Note that there
				// is no way to encode the EA's flags, since their use doesn't
				// make any sense for persisted EAs.
				hdr.Winheaders[hdrEaPrefix+ea.Name] = base64.StdEncoding.EncodeToString(ea.Value)
			}

		case winio.BackupAlternateData, winio.BackupLink, winio.BackupPropertyData, winio.BackupObjectId, winio.BackupTxfsData:
			// ignore these streams
		default:
			return fmt.Errorf("%s: unknown stream ID %d", name, bhdr.Id)
		}
	}

	err = t.WriteHeader(hdr)
	if err != nil {
		return err
	}

	if readTwice {
		// Get back to the data stream.
		if _, err = sr.Seek(restartPos, io.SeekStart); err != nil {
			return err
		}
		for dataHdr == nil {
			bhdr, err := br.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
			if bhdr.Id == winio.BackupData {
				dataHdr = bhdr
			}
		}
	}

	if dataHdr != nil {
		// A data stream was found. Copy the data.
		if (dataHdr.Attributes & winio.StreamSparseAttributes) == 0 {
			if size != dataHdr.Size {
				return fmt.Errorf("%s: mismatch between file size %d and header size %d", name, size, dataHdr.Size)
			}
			_, err = io.Copy(t, br)
			if err != nil {
				return err
			}
		} else {
			err = copySparse(t, br)
			if err != nil {
				return err
			}
		}
	}

	// Look for streams after the data stream. The only ones we handle are alternate data streams.
	// Other streams may have metadata that could be serialized, but the tar header has already
	// been written. In practice, this means that we don't get EA or TXF metadata.
	for {
		bhdr, err := br.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		switch bhdr.Id {
		case winio.BackupAlternateData:
			altName := bhdr.Name
			if strings.HasSuffix(altName, ":$DATA") {
				altName = altName[:len(altName)-len(":$DATA")]
			}
			if (bhdr.Attributes & winio.StreamSparseAttributes) == 0 {
				hdr = &tar.Header{
					Name:       name + altName,
					Mode:       hdr.Mode,
					Typeflag:   tar.TypeReg,
					Size:       bhdr.Size,
					ModTime:    hdr.ModTime,
					AccessTime: hdr.AccessTime,
					ChangeTime: hdr.ChangeTime,
				}
				err = t.WriteHeader(hdr)
				if err != nil {
					return err
				}
				_, err = io.Copy(t, br)
				if err != nil {
					return err
				}

			} else {
				// Unsupported for now, since the size of the alternate stream is not present
				// in the backup stream until after the data has been read.
				return errors.New("tar of sparse alternate data streams is unsupported")
			}
		case winio.BackupEaData, winio.BackupLink, winio.BackupPropertyData, winio.BackupObjectId, winio.BackupTxfsData:
			// ignore these streams
		default:
			return fmt.Errorf("%s: unknown stream ID %d after data", name, bhdr.Id)
		}
	}
	return nil
}

// FileInfoFromHeader retrieves basic Win32 file information from a tar header, using the additional metadata written by
// WriteTarFileFromBackupStream.
func FileInfoFromHeader(hdr *tar.Header) (name string, size int64, fileInfo *winio.FileBasicInfo, err error) {
	name = hdr.Name
	if hdr.Typeflag == tar.TypeReg || hdr.Typeflag == tar.TypeRegA {
		size = hdr.Size
	}
	fileInfo = &winio.FileBasicInfo{
		LastAccessTime: syscall.NsecToFiletime(hdr.AccessTime.UnixNano()),
		LastWriteTime:  syscall.NsecToFiletime(hdr.ModTime.UnixNano()),
		ChangeTime:     syscall.NsecToFiletime(hdr.ChangeTime.UnixNano()),
		CreationTime:   syscall.NsecToFiletime(hdr.CreationTime.UnixNano()),
	}
	if attrStr, ok := hdr.Winheaders[hdrFileAttributes]; ok {
		attr, err := strconv.ParseUint(attrStr, 10, 32)
		if err != nil {
			return "", 0, nil, err
		}
		fileInfo.FileAttributes = uintptr(attr)
	} else {
		if hdr.Typeflag == tar.TypeDir {
			fileInfo.FileAttributes |= syscall.FILE_ATTRIBUTE_DIRECTORY
		}
	}
	return
}

// WriteBackupStreamFromTarFile writes a Win32 backup stream from the current tar file. Since this function may process multiple
// tar file entries in order to collect all the alternate data streams for the file, it returns the next
// tar file that was not processed, or io.EOF is there are no more.
func WriteBackupStreamFromTarFile(w io.Writer, t *tar.Reader, hdr *tar.Header) (*tar.Header, error) {
	bw := winio.NewBackupStreamWriter(w)
	var sd []byte
	var err error
	// Maintaining old SDDL-based behavior for backward compatibility.  All new tar headers written
	// by this library will have raw binary for the security descriptor.
	if sddl, ok := hdr.Winheaders[hdrSecurityDescriptor]; ok {
		sd, err = winio.SddlToSecurityDescriptor(sddl)
		if err != nil {
			return nil, err
		}
	}
	if sdraw, ok := hdr.Winheaders[hdrRawSecurityDescriptor]; ok {
		sd, err = base64.StdEncoding.DecodeString(sdraw)
		if err != nil {
			return nil, err
		}
	}
	if len(sd) != 0 {
		bhdr := winio.BackupHeader{
			Id:   winio.BackupSecurity,
			Size: int64(len(sd)),
		}
		err := bw.WriteHeader(&bhdr)
		if err != nil {
			return nil, err
		}
		_, err = bw.Write(sd)
		if err != nil {
			return nil, err
		}
	}
	var eas []winio.ExtendedAttribute
	for k, v := range hdr.Winheaders {
		if !strings.HasPrefix(k, hdrEaPrefix) {
			continue
		}
		data, err := base64.StdEncoding.DecodeString(v)
		if err != nil {
			return nil, err
		}
		eas = append(eas, winio.ExtendedAttribute{
			Name:  k[len(hdrEaPrefix):],
			Value: data,
		})
	}
	if len(eas) != 0 {
		eadata, err := winio.EncodeExtendedAttributes(eas)
		if err != nil {
			return nil, err
		}
		bhdr := winio.BackupHeader{
			Id:   winio.BackupEaData,
			Size: int64(len(eadata)),
		}
		err = bw.WriteHeader(&bhdr)
		if err != nil {
			return nil, err
		}
		_, err = bw.Write(eadata)
		if err != nil {
			return nil, err
		}
	}
	if hdr.Typeflag == tar.TypeSymlink {
		_, isMountPoint := hdr.Winheaders[hdrMountPoint]
		rp := winio.ReparsePoint{
			Target:       filepath.FromSlash(hdr.Linkname),
			IsMountPoint: isMountPoint,
		}
		reparse := winio.EncodeReparsePoint(&rp)
		bhdr := winio.BackupHeader{
			Id:   winio.BackupReparseData,
			Size: int64(len(reparse)),
		}
		err := bw.WriteHeader(&bhdr)
		if err != nil {
			return nil, err
		}
		_, err = bw.Write(reparse)
		if err != nil {
			return nil, err
		}
	}
	if hdr.Typeflag == tar.TypeReg || hdr.Typeflag == tar.TypeRegA {
		bhdr := winio.BackupHeader{
			Id:   winio.BackupData,
			Size: hdr.Size,
		}
		err := bw.WriteHeader(&bhdr)
		if err != nil {
			return nil, err
		}
		_, err = io.Copy(bw, t)
		if err != nil {
			return nil, err
		}
	}
	// Copy all the alternate data streams and return the next non-ADS header.
	for {
		ahdr, err := t.Next()
		if err != nil {
			return nil, err
		}
		if ahdr.Typeflag != tar.TypeReg || !strings.HasPrefix(ahdr.Name, hdr.Name+":") {
			return ahdr, nil
		}
		bhdr := winio.BackupHeader{
			Id:   winio.BackupAlternateData,
			Size: ahdr.Size,
			Name: ahdr.Name[len(hdr.Name):] + ":$DATA",
		}
		err = bw.WriteHeader(&bhdr)
		if err != nil {
			return nil, err
		}
		_, err = io.Copy(bw, t)
		if err != nil {
			return nil, err
		}
	}
}
