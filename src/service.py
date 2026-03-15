import subprocess
import win32serviceutil
import win32service
import win32event
import servicemanager

class UvicornService(win32serviceutil.ServiceFramework):
    _svc_name_ = "aa-meetings"
    _svc_display_name_ = "AA Meetings API"
    _svc_description_ = "AA Meetings RAG API (uvicorn)"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.process = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        if self.process:
            self.process.terminate()
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.process = subprocess.Popen([
            r"C:\Python312\Scripts\uvicorn.exe",
            "api:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--reload"
        ], cwd=r"C:\Users\Admin\vector\src")

        self.process.wait()

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(UvicornService)