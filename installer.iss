; MatLite 数学工作台 —— Inno Setup 6 正式安装脚本
; 用法：
;   1. 运行 打包.bat（或直接 python build_exe.py）生成 dist\MatLite\
;   2. 用 Inno Setup 6 的 ISCC 编译本文件：
;        "C:\Users\19627\AppData\Local\Programs\Inno Setup 6\ISCC.exe" installer.iss
;   3. 产物：dist\MatLiteSetup.exe（正式安装包，含协议页与自定义安装选项）
; Made by zoilzo & Claude

#define MyAppName "MatLite 数学工作台"
#define MyAppVersion "1.3.0"
#define MyAppPublisher "zoilzo"
#define MyAppExeName "MatLite.exe"
#define MyAppIcon "assets\MatLite.ico"
#define MyLicenseFile "assets\installer_license.txt"

[Setup]
AppId={{9C4F6B1E-2A8D-4E5B-9C1F-0B7D3E5A8C2F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/zoilzo/MatLite
AppSupportURL=https://github.com/zoilzo/MatLite/issues
AppComments=MatLite 数学工作台 - 本地单机的数学与统计工具 - Made by zoilzo & Claude
AppCopyright=Copyright (C) 2026 zoilzo & Claude
VersionInfoVersion=1.3.0
VersionInfoDescription=MatLite 数学工作台
VersionInfoProductName=MatLite
VersionInfoProductVersion=1.3.0
DefaultGroupName=MatLite
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\Programs\MatLite
OutputDir=dist
OutputBaseFilename=MatLiteSetup
SetupIconFile={#MyAppIcon}
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UsePreviousAppDir=yes
CloseApplications=yes
RestartApplications=no
DisableDirPage=no
DisableReadyPage=no
DisableFinishedPage=no

[Languages]
; 使用随附的中文语言文件（自包含，不依赖 Inno Setup 安装目录）
Name: "chinesesimp"; MessagesFile: "assets\languages\ChineseSimplified.isl"

[Types]
Name: "full"; Description: "完整安装（推荐）"
Name: "compact"; Description: "精简安装（仅主程序，不含快捷方式）"
Name: "custom"; Description: "自定义安装（手动选择组件）"; Flags: iscustom

[Components]
Name: "main"; Description: "MatLite 主程序（必需）"; Types: full compact custom; Flags: fixed
Name: "desktopicon"; Description: "创建桌面快捷方式"; Types: full custom
Name: "startmenu"; Description: "创建「开始」菜单快捷方式"; Types: full custom

[Tasks]
Name: "autostart"; Description: "开机时自动启动 MatLite"; GroupDescription: "附加任务："; Flags: unchecked
Name: "quicklaunch"; Description: "在任务栏（快速启动）创建图标"; GroupDescription: "附加任务："; Flags: unchecked

[Files]
Source: "dist\MatLite\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Components: startmenu
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Components: desktopicon
Name: "{userappdata}\Microsoft\Windows\Start Menu\Programs\Startup\MatLite.lnk"; Filename: "{app}\{#MyAppExeName}"; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即运行 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\error_log.txt"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if MsgBox('是否同时删除本机的历史记录、账号与设置数据？' + #13#13 +
              '选择「是」将删除用户的全部本地数据；选择「否」保留这些数据。' + #13#13 +
              '建议：仅在卸载后不打算再使用时才删除数据。', mbConfirmation, MB_YESNO) = IDYES then
      DelTree(ExpandConstant('{userappdata}\MatLite'), True, True, True);
end;
