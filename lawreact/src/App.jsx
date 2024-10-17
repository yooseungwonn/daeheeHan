import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import HomeView from "./views/Homeview";

const App = () => {
  return (
      <BrowserRouter>
        <Routes>
            <Route path="/" element={<HomeView/>}/>
        </Routes>
      </BrowserRouter>
  );
};

export default App;
