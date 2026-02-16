const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("voxelomicsDesktop", {
  getAppInfo: () => ipcRenderer.invoke("desktop:get-app-info"),
  chooseDataDirectory: () => ipcRenderer.invoke("desktop:choose-data-directory"),
  saveReport: (payload) => ipcRenderer.invoke("desktop:save-report", payload),
});
