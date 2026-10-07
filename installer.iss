; installer.iss — Inno Setup 6
; Собирает WhisperGUI-Setup.exe из папки dist/WhisperGUI/

#define MyAppName        "WhisperGUI"
#define MyAppVersion     "1.0.0"
#define MyAppPublisher   "WhisperGUI Project"
#define MyAppExeName     "WhisperGUI.exe"
#define SourceDir        "dist\WhisperGUI"

[Setup]
AppId={{8F4E3F0D-3B31-4C2A-9D6A-4E86E4F1A901}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=no
OutputBaseFilename=WhisperGUI-Setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
PrivilegesRequired=lowest
SetupIconFile=installer.ico
UninstallDisplayIcon={app}\WhisperGUI.exe

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
      GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\models"
Name: "{app}\cuda"
Name: "{app}\logs"
Name: "{app}\config"
Name: "{app}\output"
Name: "{app}\output\tmp"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; \
          Flags: nowait postinstall skipifsilent

[Code]

function DirSizeMB(const Dir: string): Integer;
var
  FindRec: TFindRec;
  SubPath: string;
  Total: Int64;
begin
  Total := 0;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          SubPath := AddBackslash(Dir) + FindRec.Name;
          if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
            Total := Total + Int64(DirSizeMB(SubPath)) * 1024 * 1024
          else
            Total := Total + Int64(FindRec.SizeHigh) shl 32 + FindRec.SizeLow;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
  Result := Total div (1024 * 1024);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ModelsDir, CudaDir: string;
  ModelsMB, CudaMB: Integer;
  DelModels, DelCuda: Boolean;
begin
  if CurUninstallStep = usUninstall then
  begin
    ModelsDir := ExpandConstant('{app}\models');
    CudaDir   := ExpandConstant('{app}\cuda');

    ModelsMB := 0;
    CudaMB := 0;
    if DirExists(ModelsDir) then ModelsMB := DirSizeMB(ModelsDir);
    if DirExists(CudaDir) then CudaMB := DirSizeMB(CudaDir);

    DelModels := False;
    DelCuda := False;

    if ModelsMB > 0 then
      DelModels := MsgBox(
        'Удалить также скачанные модели Whisper? (' + IntToStr(ModelsMB) + ' МБ)',
        mbConfirmation, MB_YESNO) = IDYES;

    if CudaMB > 0 then
      DelCuda := MsgBox(
        'Удалить также CUDA-компоненты? (' + IntToStr(CudaMB) + ' МБ)',
        mbConfirmation, MB_YESNO) = IDYES;

    if DelModels then DelTree(ModelsDir, True, True, True);
    if DelCuda then DelTree(CudaDir, True, True, True);
  end;
end;