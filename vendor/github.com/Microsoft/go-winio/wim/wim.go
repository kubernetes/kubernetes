// Package wim implements a WIM file parser.
//
// WIM files are used to distribute Windows file system and container images.
// They are documented at https://msdn.microsoft.com/en-us/library/windows/desktop/dd861280.aspx.
package wim

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"strconv"
	"sync"
	"time"
	"unicode/utf16"
)

// File attribute constants from Windows.
const (
	FILE_ATTRIBUTE_READONLY            = 0x00000001
	FILE_ATTRIBUTE_HIDDEN              = 0x00000002
	FILE_ATTRIBUTE_SYSTEM              = 0x00000004
	FILE_ATTRIBUTE_DIRECTORY           = 0x00000010
	FILE_ATTRIBUTE_ARCHIVE             = 0x00000020
	FILE_ATTRIBUTE_DEVICE              = 0x00000040
	FILE_ATTRIBUTE_NORMAL              = 0x00000080
	FILE_ATTRIBUTE_TEMPORARY           = 0x00000100
	FILE_ATTRIBUTE_SPARSE_FILE         = 0x00000200
	FILE_ATTRIBUTE_REPARSE_POINT       = 0x00000400
	FILE_ATTRIBUTE_COMPRESSED          = 0x00000800
	FILE_ATTRIBUTE_OFFLINE             = 0x00001000
	FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 0x00002000
	FILE_ATTRIBUTE_ENCRYPTED           = 0x00004000
	FILE_ATTRIBUTE_INTEGRITY_STREAM    = 0x00008000
	FILE_ATTRIBUTE_VIRTUAL             = 0x00010000
	FILE_ATTRIBUTE_NO_SCRUB_DATA       = 0x00020000
	FILE_ATTRIBUTE_EA                  = 0x00040000
)

// Windows processor architectures.
const (
	PROCESSOR_ARCHITECTURE_INTEL         = 0
	PROCESSOR_ARCHITECTURE_MIPS          = 1
	PROCESSOR_ARCHITECTURE_ALPHA         = 2
	PROCESSOR_ARCHITECTURE_PPC           = 3
	PROCESSOR_ARCHITECTURE_SHX           = 4
	PROCESSOR_ARCHITECTURE_ARM           = 5
	PROCESSOR_ARCHITECTURE_IA64          = 6
	PROCESSOR_ARCHITECTURE_ALPHA64       = 7
	PROCESSOR_ARCHITECTURE_MSIL          = 8
	PROCESSOR_ARCHITECTURE_AMD64         = 9
	PROCESSOR_ARCHITECTURE_IA32_ON_WIN64 = 10
	PROCESSOR_ARCHITECTURE_NEUTRAL       = 11
	PROCESSOR_ARCHITECTURE_ARM64         = 12
)

var wimImageTag = [...]byte{'M', 'S', 'W', 'I', 'M', 0, 0, 0}

type guid struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}

func (g guid) String() string {
	return fmt.Sprintf("%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g.Data1, g.Data2, g.Data3, g.Data4[0], g.Data4[1], g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7])
}

type resourceDescriptor struct {
	FlagsAndCompressedSize uint64
	Offset                 int64
	OriginalSize           int64
}

type resFlag byte

const (
	resFlagFree resFlag = 1 << iota
	resFlagMetadata
	resFlagCompressed
	resFlagSpanned
)

const validate = false

const supportedResFlags = resFlagMetadata | resFlagCompressed

func (r *resourceDescriptor) Flags() resFlag {
	return resFlag(r.FlagsAndCompressedSize >> 56)
}

func (r *resourceDescriptor) CompressedSize() int64 {
	return int64(r.FlagsAndCompressedSize & 0xffffffffffffff)
}

func (r *resourceDescriptor) String() string {
	s := fmt.Sprintf("%d bytes at %d", r.CompressedSize(), r.Offset)
	if r.Flags()&4 != 0 {
		s += fmt.Sprintf(" (uncompresses to %d)", r.OriginalSize)
	}
	return s
}

// SHA1Hash contains the SHA1 hash of a file or stream.
type SHA1Hash [20]byte

type streamDescriptor struct {
	resourceDescriptor
	PartNumber uint16
	RefCount   uint32
	Hash       SHA1Hash
}

type hdrFlag uint32

const (
	hdrFlagReserved hdrFlag = 1 << iota
	hdrFlagCompressed
	hdrFlagReadOnly
	hdrFlagSpanned
	hdrFlagResourceOnly
	hdrFlagMetadataOnly
	hdrFlagWriteInProgress
	hdrFlagRpFix
)

const (
	hdrFlagCompressReserved hdrFlag = 1 << (iota + 16)
	hdrFlagCompressXpress
	hdrFlagCompressLzx
)

const supportedHdrFlags = hdrFlagRpFix | hdrFlagReadOnly | hdrFlagCompressed | hdrFlagCompressLzx

type wimHeader struct {
	ImageTag        [8]byte
	Size            uint32
	Version         uint32
	Flags           hdrFlag
	CompressionSize uint32
	WIMGuid         guid
	PartNumber      uint16
	TotalParts      uint16
	ImageCount      uint32
	OffsetTable     resourceDescriptor
	XMLData         resourceDescriptor
	BootMetadata    resourceDescriptor
	BootIndex       uint32
	Padding         uint32
	Integrity       resourceDescriptor
	Unused          [60]byte
}

type securityblockDisk struct {
	TotalLength uint32
	NumEntries  uint32
}

const securityblockDiskSize = 8

type direntry struct {
	Attributes       uint32
	SecurityID       uint32
	SubdirOffset     int64
	Unused1, Unused2 int64
	CreationTime     Filetime
	LastAccessTime   Filetime
	LastWriteTime    Filetime
	Hash             SHA1Hash
	Padding          uint32
	ReparseHardLink  int64
	StreamCount      uint16
	ShortNameLength  uint16
	FileNameLength   uint16
}

var direntrySize = int64(binary.Size(direntry{}) + 8) // includes an 8-byte length prefix

type streamentry struct {
	Unused     int64
	Hash       SHA1Hash
	NameLength int16
}

var streamentrySize = int64(binary.Size(streamentry{}) + 8) // includes an 8-byte length prefix

// Filetime represents a Windows time.
type Filetime struct {
	LowDateTime  uint32
	HighDateTime uint32
}

// Time returns the time as time.Time.
func (ft *Filetime) Time() time.Time {
	// 100-nanosecond intervals since January 1, 1601
	nsec := int64(ft.HighDateTime)<<32 + int64(ft.LowDateTime)
	// change starting time to the Epoch (00:00:00 UTC, January 1, 1970)
	nsec -= 116444736000000000
	// convert into nanoseconds
	nsec *= 100
	return time.Unix(0, nsec)
}

// UnmarshalXML unmarshals the time from a WIM XML blob.
func (ft *Filetime) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	type time struct {
		Low  string `xml:"LOWPART"`
		High string `xml:"HIGHPART"`
	}
	var t time
	err := d.DecodeElement(&t, &start)
	if err != nil {
		return err
	}

	low, err := strconv.ParseUint(t.Low, 0, 32)
	if err != nil {
		return err
	}
	high, err := strconv.ParseUint(t.High, 0, 32)
	if err != nil {
		return err
	}

	ft.LowDateTime = uint32(low)
	ft.HighDateTime = uint32(high)
	return nil
}

type info struct {
	Image []ImageInfo `xml:"IMAGE"`
}

// ImageInfo contains information about the image.
type ImageInfo struct {
	Name         string       `xml:"NAME"`
	Index        int          `xml:"INDEX,attr"`
	CreationTime Filetime     `xml:"CREATIONTIME"`
	ModTime      Filetime     `xml:"LASTMODIFICATIONTIME"`
	Windows      *WindowsInfo `xml:"WINDOWS"`
}

// WindowsInfo contains information about the Windows installation in the image.
type WindowsInfo struct {
	Arch             byte     `xml:"ARCH"`
	ProductName      string   `xml:"PRODUCTNAME"`
	EditionID        string   `xml:"EDITIONID"`
	InstallationType string   `xml:"INSTALLATIONTYPE"`
	ProductType      string   `xml:"PRODUCTTYPE"`
	Languages        []string `xml:"LANGUAGES>LANGUAGE"`
	DefaultLanguage  string   `xml:"LANGUAGES>DEFAULT"`
	Version          Version  `xml:"VERSION"`
	SystemRoot       string   `xml:"SYSTEMROOT"`
}

// Version represents a Windows build version.
type Version struct {
	Major   int `xml:"MAJOR"`
	Minor   int `xml:"MINOR"`
	Build   int `xml:"BUILD"`
	SPBuild int `xml:"SPBUILD"`
	SPLevel int `xml:"SPLEVEL"`
}

// ParseError is returned when the WIM cannot be parsed.
type ParseError struct {
	Oper string
	Path string
	Err  error
}

func (e *ParseError) Error() string {
	if e.Path == "" {
		return "WIM parse error at " + e.Oper + ": " + e.Err.Error()
	}
	return fmt.Sprintf("WIM parse error: %s %s: %s", e.Oper, e.Path, e.Err.Error())
}

// Reader provides functions to read a WIM file.
type Reader struct {
	hdr      wimHeader
	r        io.ReaderAt
	fileData map[SHA1Hash]resourceDescriptor

	XMLInfo string   // The XML information about the WIM.
	Image   []*Image // The WIM's images.
}

// Image represents an image within a WIM file.
type Image struct {
	wim        *Reader
	offset     resourceDescriptor
	sds        [][]byte
	rootOffset int64
	r          io.ReadCloser
	curOffset  int64
	m          sync.Mutex

	ImageInfo
}

// StreamHeader contains alternate data stream metadata.
type StreamHeader struct {
	Name string
	Hash SHA1Hash
	Size int64
}

// Stream represents an alternate data stream or reparse point data stream.
type Stream struct {
	StreamHeader
	wim    *Reader
	offset resourceDescriptor
}

// FileHeader contains file metadata.
type FileHeader struct {
	Name               string
	ShortName          string
	Attributes         uint32
	SecurityDescriptor []byte
	CreationTime       Filetime
	LastAccessTime     Filetime
	LastWriteTime      Filetime
	Hash               SHA1Hash
	Size               int64
	LinkID             int64
	ReparseTag         uint32
	ReparseReserved    uint32
}

// File represents a file or directory in a WIM image.
type File struct {
	FileHeader
	Streams      []*Stream
	offset       resourceDescriptor
	img          *Image
	subdirOffset int64
}

// NewReader returns a Reader that can be used to read WIM file data.
func NewReader(f io.ReaderAt) (*Reader, error) {
	r := &Reader{r: f}
	section := io.NewSectionReader(f, 0, 0xffff)
	err := binary.Read(section, binary.LittleEndian, &r.hdr)
	if err != nil {
		return nil, err
	}

	if r.hdr.ImageTag != wimImageTag {
		return nil, &ParseError{Oper: "image tag", Err: errors.New("not a WIM file")}
	}

	if r.hdr.Flags&^supportedHdrFlags != 0 {
		return nil, fmt.Errorf("unsupported WIM flags %x", r.hdr.Flags&^supportedHdrFlags)
	}

	if r.hdr.CompressionSize != 0x8000 {
		return nil, fmt.Errorf("unsupported compression size %d", r.hdr.CompressionSize)
	}

	if r.hdr.TotalParts != 1 {
		return nil, errors.New("multi-part WIM not supported")
	}

	fileData, images, err := r.readOffsetTable(&r.hdr.OffsetTable)
	if err != nil {
		return nil, err
	}

	xmlinfo, err := r.readXML()
	if err != nil {
		return nil, err
	}

	var info info
	err = xml.Unmarshal([]byte(xmlinfo), &info)
	if err != nil {
		return nil, &ParseError{Oper: "XML info", Err: err}
	}

	for i, img := range images {
		for _, imgInfo := range info.Image {
			if imgInfo.Index == i+1 {
				img.ImageInfo = imgInfo
				break
			}
		}
	}

	r.fileData = fileData
	r.Image = images
	r.XMLInfo = xmlinfo
	return r, nil
}

// Close releases resources associated with the Reader.
func (r *Reader) Close() error {
	for _, img := range r.Image {
		img.reset()
	}
	return nil
}

func (r *Reader) resourceReader(hdr *resourceDescriptor) (io.ReadCloser, error) {
	return r.resourceReaderWithOffset(hdr, 0)
}

func (r *Reader) resourceReaderWithOffset(hdr *resourceDescriptor, offset int64) (io.ReadCloser, error) {
	var sr io.ReadCloser
	section := io.NewSectionReader(r.r, hdr.Offset, hdr.CompressedSize())
	if hdr.Flags()&resFlagCompressed == 0 {
		section.Seek(offset, 0)
		sr = ioutil.NopCloser(section)
	} else {
		cr, err := newCompressedReader(section, hdr.OriginalSize, offset)
		if err != nil {
			return nil, err
		}
		sr = cr
	}

	return sr, nil
}

func (r *Reader) readResource(hdr *resourceDescriptor) ([]byte, error) {
	rsrc, err := r.resourceReader(hdr)
	if err != nil {
		return nil, err
	}
	defer rsrc.Close()
	return ioutil.ReadAll(rsrc)
}

func (r *Reader) readXML() (string, error) {
	if r.hdr.XMLData.CompressedSize() == 0 {
		return "", nil
	}
	rsrc, err := r.resourceReader(&r.hdr.XMLData)
	if err != nil {
		return "", err
	}
	defer rsrc.Close()

	XMLData := make([]uint16, r.hdr.XMLData.OriginalSize/2)
	err = binary.Read(rsrc, binary.LittleEndian, XMLData)
	if err != nil {
		return "", &ParseError{Oper: "XML data", Err: err}
	}

	// The BOM will always indicate little-endian UTF-16.
	if XMLData[0] != 0xfeff {
		return "", &ParseError{Oper: "XML data", Err: errors.New("invalid BOM")}
	}
	return string(utf16.Decode(XMLData[1:])), nil
}

func (r *Reader) readOffsetTable(res *resourceDescriptor) (map[SHA1Hash]resourceDescriptor, []*Image, error) {
	fileData := make(map[SHA1Hash]resourceDescriptor)
	var images []*Image

	offsetTable, err := r.readResource(res)
	if err != nil {
		return nil, nil, &ParseError{Oper: "offset table", Err: err}
	}

	br := bytes.NewReader(offsetTable)
	for i := 0; ; i++ {
		var res streamDescriptor
		err := binary.Read(br, binary.LittleEndian, &res)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, &ParseError{Oper: "offset table", Err: err}
		}
		if res.Flags()&^supportedResFlags != 0 {
			return nil, nil, &ParseError{Oper: "offset table", Err: errors.New("unsupported resource flag")}
		}

		// Validation for ad-hoc testing
		if validate {
			sec, err := r.resourceReader(&res.resourceDescriptor)
			if err != nil {
				panic(fmt.Sprint(i, err))
			}
			hash := sha1.New()
			_, err = io.Copy(hash, sec)
			sec.Close()
			if err != nil {
				panic(fmt.Sprint(i, err))
			}
			var cmphash SHA1Hash
			copy(cmphash[:], hash.Sum(nil))
			if cmphash != res.Hash {
				panic(fmt.Sprint(i, "hash mismatch"))
			}
		}

		if res.Flags()&resFlagMetadata != 0 {
			image := &Image{
				wim:    r,
				offset: res.resourceDescriptor,
			}
			images = append(images, image)
		} else {
			fileData[res.Hash] = res.resourceDescriptor
		}
	}

	if len(images) != int(r.hdr.ImageCount) {
		return nil, nil, &ParseError{Oper: "offset table", Err: errors.New("mismatched image count")}
	}

	return fileData, images, nil
}

func (r *Reader) readSecurityDescriptors(rsrc io.Reader) (sds [][]byte, n int64, err error) {
	var secBlock securityblockDisk
	err = binary.Read(rsrc, binary.LittleEndian, &secBlock)
	if err != nil {
		err = &ParseError{Oper: "security table", Err: err}
		return
	}

	n += securityblockDiskSize

	secSizes := make([]int64, secBlock.NumEntries)
	err = binary.Read(rsrc, binary.LittleEndian, &secSizes)
	if err != nil {
		err = &ParseError{Oper: "security table sizes", Err: err}
		return
	}

	n += int64(secBlock.NumEntries * 8)

	sds = make([][]byte, secBlock.NumEntries)
	for i, size := range secSizes {
		sd := make([]byte, size&0xffffffff)
		_, err = io.ReadFull(rsrc, sd)
		if err != nil {
			err = &ParseError{Oper: "security descriptor", Err: err}
			return
		}
		n += int64(len(sd))
		sds[i] = sd
	}

	secsize := int64((secBlock.TotalLength + 7) &^ 7)
	if n > secsize {
		err = &ParseError{Oper: "security descriptor", Err: errors.New("security descriptor table too small")}
		return
	}

	_, err = io.CopyN(ioutil.Discard, rsrc, secsize-n)
	if err != nil {
		return
	}

	n = secsize
	return
}

// Open parses the image and returns the root directory.
func (img *Image) Open() (*File, error) {
	if img.sds == nil {
		rsrc, err := img.wim.resourceReaderWithOffset(&img.offset, img.rootOffset)
		if err != nil {
			return nil, err
		}
		sds, n, err := img.wim.readSecurityDescriptors(rsrc)
		if err != nil {
			rsrc.Close()
			return nil, err
		}
		img.sds = sds
		img.r = rsrc
		img.rootOffset = n
		img.curOffset = n
	}

	f, err := img.readdir(img.rootOffset)
	if err != nil {
		return nil, err
	}
	if len(f) != 1 {
		return nil, &ParseError{Oper: "root directory", Err: errors.New("expected exactly 1 root directory entry")}
	}
	return f[0], err
}

func (img *Image) reset() {
	if img.r != nil {
		img.r.Close()
		img.r = nil
	}
	img.curOffset = -1
}

func (img *Image) readdir(offset int64) ([]*File, error) {
	img.m.Lock()
	defer img.m.Unlock()

	if offset < img.curOffset || offset > img.curOffset+chunkSize {
		// Reset to seek backward or to seek forward very far.
		img.reset()
	}
	if img.r == nil {
		rsrc, err := img.wim.resourceReaderWithOffset(&img.offset, offset)
		if err != nil {
			return nil, err
		}
		img.r = rsrc
		img.curOffset = offset
	}
	if offset > img.curOffset {
		_, err := io.CopyN(ioutil.Discard, img.r, offset-img.curOffset)
		if err != nil {
			img.reset()
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return nil, err
		}
	}

	var entries []*File
	for {
		e, n, err := img.readNextEntry(img.r)
		img.curOffset += n
		if err == io.EOF {
			break
		}
		if err != nil {
			img.reset()
			return nil, err
		}
		entries = append(entries, e)
	}
	return entries, nil
}

func (img *Image) readNextEntry(r io.Reader) (*File, int64, error) {
	var length int64
	err := binary.Read(r, binary.LittleEndian, &length)
	if err != nil {
		return nil, 0, &ParseError{Oper: "directory length check", Err: err}
	}

	if length == 0 {
		return nil, 8, io.EOF
	}

	left := length
	if left < direntrySize {
		return nil, 0, &ParseError{Oper: "directory entry", Err: errors.New("size too short")}
	}

	var dentry direntry
	err = binary.Read(r, binary.LittleEndian, &dentry)
	if err != nil {
		return nil, 0, &ParseError{Oper: "directory entry", Err: err}
	}

	left -= direntrySize

	namesLen := int64(dentry.FileNameLength + 2 + dentry.ShortNameLength)
	if left < namesLen {
		return nil, 0, &ParseError{Oper: "directory entry", Err: errors.New("size too short for names")}
	}

	names := make([]uint16, namesLen/2)
	err = binary.Read(r, binary.LittleEndian, names)
	if err != nil {
		return nil, 0, &ParseError{Oper: "file name", Err: err}
	}

	left -= namesLen

	var name, shortName string
	if dentry.FileNameLength > 0 {
		name = string(utf16.Decode(names[:dentry.FileNameLength/2]))
	}

	if dentry.ShortNameLength > 0 {
		shortName = string(utf16.Decode(names[dentry.FileNameLength/2+1:]))
	}

	var offset resourceDescriptor
	zerohash := SHA1Hash{}
	if dentry.Hash != zerohash {
		var ok bool
		offset, ok = img.wim.fileData[dentry.Hash]
		if !ok {
			return nil, 0, &ParseError{Oper: "directory entry", Path: name, Err: fmt.Errorf("could not find file data matching hash %#v", dentry)}
		}
	}

	f := &File{
		FileHeader: FileHeader{
			Attributes:     dentry.Attributes,
			CreationTime:   dentry.CreationTime,
			LastAccessTime: dentry.LastAccessTime,
			LastWriteTime:  dentry.LastWriteTime,
			Hash:           dentry.Hash,
			Size:           offset.OriginalSize,
			Name:           name,
			ShortName:      shortName,
		},

		offset:       offset,
		img:          img,
		subdirOffset: dentry.SubdirOffset,
	}

	isDir := false

	if dentry.Attributes&FILE_ATTRIBUTE_REPARSE_POINT == 0 {
		f.LinkID = dentry.ReparseHardLink
		if dentry.Attributes&FILE_ATTRIBUTE_DIRECTORY != 0 {
			isDir = true
		}
	} else {
		f.ReparseTag = uint32(dentry.ReparseHardLink)
		f.ReparseReserved = uint32(dentry.ReparseHardLink >> 32)
	}

	if isDir && f.subdirOffset == 0 {
		return nil, 0, &ParseError{Oper: "directory entry", Path: name, Err: errors.New("no subdirectory data for directory")}
	} else if !isDir && f.subdirOffset != 0 {
		return nil, 0, &ParseError{Oper: "directory entry", Path: name, Err: errors.New("unexpected subdirectory data for non-directory")}
	}

	if dentry.SecurityID != 0xffffffff {
		f.SecurityDescriptor = img.sds[dentry.SecurityID]
	}

	_, err = io.CopyN(ioutil.Discard, r, left)
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, 0, err
	}

	if dentry.StreamCount > 0 {
		var streams []*Stream
		for i := uint16(0); i < dentry.StreamCount; i++ {
			s, n, err := img.readNextStream(r)
			length += n
			if err != nil {
				return nil, 0, err
			}
			// The first unnamed stream should be treated as the file stream.
			if i == 0 && s.Name == "" {
				f.Hash = s.Hash
				f.Size = s.Size
				f.offset = s.offset
			} else if s.Name != "" {
				streams = append(streams, s)
			}
		}
		f.Streams = streams
	}

	if dentry.Attributes&FILE_ATTRIBUTE_REPARSE_POINT != 0 && f.Size == 0 {
		return nil, 0, &ParseError{Oper: "directory entry", Path: name, Err: errors.New("reparse point is missing reparse stream")}
	}

	return f, length, nil
}

func (img *Image) readNextStream(r io.Reader) (*Stream, int64, error) {
	var length int64
	err := binary.Read(r, binary.LittleEndian, &length)
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, 0, &ParseError{Oper: "stream length check", Err: err}
	}

	left := length
	if left < streamentrySize {
		return nil, 0, &ParseError{Oper: "stream entry", Err: errors.New("size too short")}
	}

	var sentry streamentry
	err = binary.Read(r, binary.LittleEndian, &sentry)
	if err != nil {
		return nil, 0, &ParseError{Oper: "stream entry", Err: err}
	}

	left -= streamentrySize

	if left < int64(sentry.NameLength) {
		return nil, 0, &ParseError{Oper: "stream entry", Err: errors.New("size too short for name")}
	}

	names := make([]uint16, sentry.NameLength/2)
	err = binary.Read(r, binary.LittleEndian, names)
	if err != nil {
		return nil, 0, &ParseError{Oper: "file name", Err: err}
	}

	left -= int64(sentry.NameLength)
	name := string(utf16.Decode(names))

	var offset resourceDescriptor
	if sentry.Hash != (SHA1Hash{}) {
		var ok bool
		offset, ok = img.wim.fileData[sentry.Hash]
		if !ok {
			return nil, 0, &ParseError{Oper: "stream entry", Path: name, Err: fmt.Errorf("could not find file data matching hash %v", sentry.Hash)}
		}
	}

	s := &Stream{
		StreamHeader: StreamHeader{
			Hash: sentry.Hash,
			Size: offset.OriginalSize,
			Name: name,
		},
		wim:    img.wim,
		offset: offset,
	}

	_, err = io.CopyN(ioutil.Discard, r, left)
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, 0, err
	}

	return s, length, nil
}

// Open returns an io.ReadCloser that can be used to read the stream's contents.
func (s *Stream) Open() (io.ReadCloser, error) {
	return s.wim.resourceReader(&s.offset)
}

// Open returns an io.ReadCloser that can be used to read the file's contents.
func (f *File) Open() (io.ReadCloser, error) {
	return f.img.wim.resourceReader(&f.offset)
}

// Readdir reads the directory entries.
func (f *File) Readdir() ([]*File, error) {
	if !f.IsDir() {
		return nil, errors.New("not a directory")
	}
	return f.img.readdir(f.subdirOffset)
}

// IsDir returns whether the given file is a directory. It returns false when it
// is a directory reparse point.
func (f *FileHeader) IsDir() bool {
	return f.Attributes&(FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_REPARSE_POINT) == FILE_ATTRIBUTE_DIRECTORY
}
