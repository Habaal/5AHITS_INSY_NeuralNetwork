@echo off
REM Baut Teil 4 und 5. Nimmt gcc, wenn vorhanden, sonst den Zig-C-Compiler (pip install ziglang).
cd /d "%~dp0"
where gcc >nul 2>nul
if %errorlevel%==0 (set CC=gcc) else (set CC=python -m ziglang cc)

%CC% -std=c99 -O2 -Wall -Wextra -o part4_txt.exe part4_txt.c nn.c -lm || exit /b 1
%CC% -std=c99 -O2 -Wall -Wextra -I. -I../output -o part5_header.exe part5_header.c nn.c -lm || exit /b 1
echo Fertig: part4_txt.exe und part5_header.exe
