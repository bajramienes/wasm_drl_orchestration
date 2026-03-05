@echo off
setlocal enabledelayedexpansion

:: List of DRL agents
set agents=hydra tdmpc2 costar dreamerv3 baseline

:: Loop through agents and create containers
for %%A in (%agents%) do (
    echo Creating container for %%A...
    docker rm -f %%A >nul 2>&1
    docker run -dit --name %%A -v "%cd%":/app -w /app python:3.10-slim tail -f /dev/null
)

echo All containers created successfully.
pause
