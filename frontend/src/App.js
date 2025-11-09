import React, { useState, useEffect, useMemo } from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { 
  Container, Toolbar, Typography, Box, CssBaseline, ThemeProvider, 
  createTheme, useMediaQuery, Drawer, List, ListItem, 
  ListItemIcon, ListItemText, Divider, Fade, 
  Menu, MenuItem, Switch, Badge,
  Snackbar, Alert, LinearProgress, Button
} from '@mui/material';
import { styled } from '@mui/material/styles';
import Dashboard from './components/Dashboard';
import CaseList from './components/CaseList';
import CaseDetail from './components/CaseDetail';
import PersonDetail from './components/PersonDetail';
import TimelineView from './components/TimelineView';
import UploadForm from './components/UploadForm';
import CaseChat from './components/CaseChat';
import LegalAssistantChat from './components/LegalAssistantChat';
import UnifiedChat from './components/UnifiedChat';

// Icons
import DashboardIcon from '@mui/icons-material/Dashboard';
import FolderIcon from '@mui/icons-material/Folder';
import SearchIcon from '@mui/icons-material/Search';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SettingsIcon from '@mui/icons-material/Settings';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import SpeedIcon from '@mui/icons-material/Speed';
import ChatIcon from '@mui/icons-material/Chat';
import GavelIcon from '@mui/icons-material/Gavel';
import VideoPlayer from './components/VideoPlayer';
import HomePage from './components/HomePage';

//NEW
import ChatDashboard from './components/ChatDashboard';

// Create themes with dark mode support and modern design
const createAppTheme = (mode) => createTheme({
  palette: {
    mode,
    primary: {
      main: mode === 'light' ? '#3b82f6' : '#60a5fa', // Modern blue
      light: mode === 'light' ? '#60a5fa' : '#93c5fd',
      dark: mode === 'light' ? '#2563eb' : '#3b82f6',
      contrastText: '#ffffff',
    },
    secondary: {
      main: mode === 'light' ? '#8b5cf6' : '#a78bfa', // Modern purple
      light: mode === 'light' ? '#a78bfa' : '#c4b5fd',
      dark: mode === 'light' ? '#7c3aed' : '#8b5cf6',
      contrastText: '#ffffff',
    },
    background: {
      default: mode === 'light' ? '#f8fafc' : '#0a0e1a',
      paper: mode === 'light' ? '#ffffff' : '#1a1f2e',
    },
    text: {
      primary: mode === 'light' ? '#1e293b' : '#f1f5f9',
      secondary: mode === 'light' ? '#64748b' : '#94a3b8',
    },
    success: {
      main: mode === 'light' ? '#10b981' : '#34d399',
      light: '#6ee7b7',
      dark: '#059669',
    },
    warning: {
      main: mode === 'light' ? '#f59e0b' : '#fbbf24',
      light: '#fde047',
      dark: '#d97706',
    },
    error: {
      main: mode === 'light' ? '#ef4444' : '#f87171',
      light: '#fca5a5',
      dark: '#dc2626',
    },
    info: {
      main: mode === 'light' ? '#0ea5e9' : '#38bdf8',
      light: '#7dd3fc',
      dark: '#0284c7',
    },
    divider: mode === 'light' ? '#e2e8f0' : '#334155',
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h4: {
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h5: {
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    button: {
      fontWeight: 600,
      letterSpacing: '0.02em',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: 'none',
          borderBottom: mode === 'light' ? '1px solid #e2e8f0' : '1px solid #334155',
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          padding: '10px 20px',
          borderRadius: '10px',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: mode === 'light' 
              ? '0 10px 15px -3px rgb(0 0 0 / 0.1)' 
              : '0 10px 15px -3px rgb(0 0 0 / 0.4)',
          },
        },
        contained: {
          boxShadow: mode === 'light'
            ? '0 4px 6px -1px rgb(0 0 0 / 0.1)'
            : '0 4px 6px -1px rgb(0 0 0 / 0.3)',
        },
        outlined: {
          borderWidth: '2px',
          '&:hover': {
            borderWidth: '2px',
          },
        },
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: mode === 'light'
            ? '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)'
            : '0 10px 15px -3px rgb(0 0 0 / 0.3), 0 4px 6px -4px rgb(0 0 0 / 0.3)',
          borderRadius: '16px',
          border: mode === 'light' ? '1px solid #f1f5f9' : '1px solid #1e293b',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: mode === 'light'
              ? '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 10px 10px -5px rgb(0 0 0 / 0.04)'
              : '0 20px 25px -5px rgb(0 0 0 / 0.5), 0 10px 10px -5px rgb(0 0 0 / 0.2)',
          },
        }
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '14px',
          backgroundImage: 'none',
          border: mode === 'light' ? '1px solid #f1f5f9' : '1px solid #1e293b',
        },
        elevation1: {
          boxShadow: mode === 'light'
            ? '0 1px 3px 0 rgb(0 0 0 / 0.1)'
            : '0 4px 6px -1px rgb(0 0 0 / 0.3)',
        },
        elevation2: {
          boxShadow: mode === 'light'
            ? '0 4px 6px -1px rgb(0 0 0 / 0.1)'
            : '0 10px 15px -3px rgb(0 0 0 / 0.3)',
        },
        elevation3: {
          boxShadow: mode === 'light'
            ? '0 10px 15px -3px rgb(0 0 0 / 0.1)'
            : '0 20px 25px -5px rgb(0 0 0 / 0.4)',
        },
      }
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: '10px',
          marginBottom: '4px',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateX(4px)',
          },
        }
      }
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          fontWeight: 600,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '10px',
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              boxShadow: mode === 'light'
                ? '0 0 0 3px rgb(59 130 246 / 0.1)'
                : '0 0 0 3px rgb(96 165 250 / 0.2)',
            },
            '&.Mui-focused': {
              boxShadow: mode === 'light'
                ? '0 0 0 3px rgb(59 130 246 / 0.2)'
                : '0 0 0 3px rgb(96 165 250 / 0.3)',
            },
          },
        },
      },
    },
    MuiContainer: {
      styleOverrides: {
        root: {
          paddingLeft: '16px',
          paddingRight: '16px',
          '@media (min-width: 600px)': {
            paddingLeft: '24px',
            paddingRight: '24px',
          },
        }
      }
    }
  },
});

// Styled drawer
const drawerWidth = 250;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })(
  ({ theme, open }) => ({
    flexGrow: 1,
    transition: theme.transitions.create(['margin', 'width'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    marginLeft: 0,
    ...(open && {
      transition: theme.transitions.create(['margin', 'width'], {
        easing: theme.transitions.easing.easeOut,
        duration: theme.transitions.duration.enteringScreen,
      }),
      [theme.breakpoints.up('md')]: {
        marginLeft: 0,
        width: '100%',
      },
    }),
  }),
);

// Main App component
function App() {
  const [useDarkMode, setUseDarkMode] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true); 
  const isMobile = useMediaQuery('(max-width:900px)');
  const [isLoading, setIsLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  
  // Create theme based on dark mode setting
  const theme = useMemo(() => createAppTheme(useDarkMode ? 'dark' : 'light'), [useDarkMode]);
  
  // Settings menu state
  const [settingsAnchorEl, setSettingsAnchorEl] = useState(null);
  
  // Notification menu state
  const [notifAnchorEl, setNotifAnchorEl] = useState(null);
  
  // Handle settings menu
  const handleSettingsClose = () => {
    setSettingsAnchorEl(null);
  };
  
  // Handle notification menu
  const handleNotifClose = () => {
    setNotifAnchorEl(null);
  };
  
  // Toggle dark mode
  const handleToggleDarkMode = () => {
    setIsLoading(true);
    // Simulate a loading time for the theme switch
    setTimeout(() => {
      setUseDarkMode(!useDarkMode);
      setIsLoading(false);
      setSnackbar({
        open: true, 
        message: !useDarkMode ? 'Dark mode activated' : 'Light mode activated', 
        severity: 'success'
      });
    }, 400);
    handleSettingsClose();
  };
  
  // Close snackbar
  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') return;
    setSnackbar({ ...snackbar, open: false });
  };
  
  useEffect(() => {
    // Initial drawer state based on screen size
    if (isMobile) {
      setDrawerOpen(false);
    }
  }, [isMobile]);

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };
  
  // Mock notifications data
  const mockNotifications = [
    { id: 1, title: "New incident uploaded", description: "Incident #T2023-0542 has been processed", time: "10 min ago", read: false },
    { id: 2, title: "Processing complete", description: "Video analysis completed for Incident #T2023-0539", time: "1 hour ago", read: false },
    { id: 3, title: "System update available", description: "AutoVision v2.1.0 is ready to install", time: "2 hours ago", read: false },
  ];
  
  // Simulate page transition loading
  const simulatePageLoading = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 700);
  };
  
  const showSnackbarMessage = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };
  
  // Navigation items - Unified AI Assistant
    const drawerItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard', badge: 0 },
    { text: 'Incidents', icon: <FolderIcon />, path: '/incidents', badge: 0 },
    { text: 'AI Assistant', icon: <GavelIcon />, path: '/assistant', badge: 0 },
    { text: 'Upload', icon: <CloudUploadIcon />, path: '/upload', badge: 0 },
  ];
  
  // Navigation link component with active state
  const NavLink = ({ to, icon, text, badge, onClick }) => {
    const location = useLocation();
    const isActive = location.pathname === to || 
      (to !== '/' && location.pathname.startsWith(to));
    
    return (
      <ListItem 
        button 
        component={Link} 
        to={to}
        onClick={() => {
          onClick && onClick();
          simulatePageLoading();
        }}
        sx={{
          borderRadius: '12px',
          mb: 0.75,
          mx: 1.5,
          px: 1.5,
          py: 1,
          bgcolor: isActive 
            ? 'primary.main'
            : 'transparent',
          color: isActive ? 'white' : 'inherit',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          boxShadow: isActive 
            ? '0 4px 12px rgba(59, 130, 246, 0.3)'
            : 'none',
          '&:hover': {
            bgcolor: isActive 
              ? 'primary.dark'
              : 'action.hover',
            transform: 'translateX(6px)',
            boxShadow: isActive
              ? '0 6px 16px rgba(59, 130, 246, 0.4)'
              : '0 2px 8px rgba(0, 0, 0, 0.1)',
          },
        }}
      >
        <ListItemIcon sx={{ color: isActive ? 'white' : 'text.secondary', minWidth: 40 }}>
          {badge ? (
            <Badge badgeContent={badge} color="error" variant="dot">
              {icon}
            </Badge>
          ) : icon}
        </ListItemIcon>
        <ListItemText 
          primary={text} 
          primaryTypographyProps={{ 
            fontWeight: isActive ? 600 : 500,
            fontSize: '0.95rem',
          }} 
        />
      </ListItem>
    );
  };
  
  const drawer = (
    <>
      {/* Navigation menu */}
      <Box 
        sx={{ 
          pt: 1, 
          pb: 2, 
          height: isMobile ? 'auto' : 'calc(100vh - 340px)'
        }}
      >
        <List component="nav" disablePadding>
          <ListItem
            button
            component={Link}
            to="/"
            onClick={isMobile ? handleDrawerToggle : undefined}
            sx={{
              py: 1.5,
              px: 3,
              mt: 2,
              mb: 1,
              justifyContent: 'center',
              '&:hover': {
                bgcolor: theme.palette.mode === 'dark' 
                  ? 'rgba(255,255,255,0.05)' 
                  : 'rgba(0,0,0,0.04)',
              }
            }}
          >
            <ListItemText 
              primary="Autovision" 
              primaryTypographyProps={{ 
                fontWeight: 600,
                fontSize: '1.3rem',
                textAlign: 'center',
              }} 
            />
          </ListItem>
        </List>
        
        <Typography
          variant="overline"
          sx={{
            display: 'block',
            px: 3,
            color: 'text.secondary',
            fontWeight: 600,
            fontSize: '0.7rem',
            letterSpacing: 1,
            mb: 0.5,
            mt: 2
          }}
        >
          MAIN MENU
        </Typography>
        
        <List component="nav" disablePadding>
          {drawerItems.map((item) => (
            <NavLink 
              key={item.text}
              to={item.path}
              icon={item.icon}
              text={item.text}
              badge={item.badge}
              onClick={isMobile ? handleDrawerToggle : undefined}
            />
          ))}
        </List>
        
        <Box sx={{ mt: 2 }}>
          <Typography 
            variant="overline" 
            sx={{ 
              display: 'block',
              px: 3, 
              color: 'text.secondary', 
              fontWeight: 600,
              fontSize: '0.7rem',
              letterSpacing: 1,
              mb: 0.5
            }}
          >
          </Typography>
          
          <List component="nav" disablePadding>
        
          </List>
        </Box>
      </Box>
      
      {/* Bottom section with dark mode toggle */}
      <Box sx={{ 
        mt: 'auto', 
        p: 2, 
        borderTop: `1px solid ${theme.palette.divider}`,
      }}>
        <Box sx={{ 
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          p: 1,
          borderRadius: 2,
        }}>
          <Typography variant="body2">Dark Mode</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <LightModeIcon 
              fontSize="small" 
              color={useDarkMode ? 'disabled' : 'warning'} 
              sx={{ opacity: useDarkMode ? 0.5 : 1 }}
            />
            <Switch 
              checked={useDarkMode} 
              onChange={handleToggleDarkMode}
              size="small"
              color="primary"
            />
            <DarkModeIcon 
              fontSize="small" 
              color={useDarkMode ? 'primary' : 'disabled'} 
              sx={{ opacity: useDarkMode ? 1 : 0.5 }}
            />
          </Box>
        </Box>
      
        
     
      </Box>
    </>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router future={{ 
        v7_startTransition: true,
        v7_relativeSplatPath: true 
      }}>
        <Box sx={{ 
          display: 'flex', 
          minHeight: '100vh',
          bgcolor: theme.palette.background.default,
        }}>
          {/* Global loading indicator */}
          {isLoading && (
            <LinearProgress 
              sx={{ 
                position: 'fixed', 
                top: 0, 
                left: 0, 
                right: 0, 
                zIndex: theme.zIndex.drawer + 2,
                height: 3
              }} 
            />
          )}

          {/* Settings menu */}
          <Menu
            anchorEl={settingsAnchorEl}
            open={Boolean(settingsAnchorEl)}
            onClose={handleSettingsClose}
            PaperProps={{
              elevation: 3,
              sx: { 
                mt: 1.5, 
                borderRadius: 2,
                minWidth: 200,
                overflow: 'visible',
                '&:before': {
                  content: '""',
                  display: 'block',
                  position: 'absolute',
                  top: 0,
                  right: 14,
                  width: 10,
                  height: 10,
                  bgcolor: 'background.paper',
                  transform: 'translateY(-50%) rotate(45deg)',
                  zIndex: 0,
                },
              },
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <MenuItem onClick={handleToggleDarkMode} sx={{ gap: 2 }}>
              {useDarkMode ? (
                <><LightModeIcon fontSize="small" /> Light Mode</>
              ) : (
                <><DarkModeIcon fontSize="small" /> Dark Mode</>
              )}
            </MenuItem>
            <MenuItem sx={{ gap: 2 }}>
              <SpeedIcon fontSize="small" /> Performance
            </MenuItem>
            <MenuItem sx={{ gap: 2 }}>
              <SettingsIcon fontSize="small" /> Preferences
            </MenuItem>
            <Divider />
            <MenuItem sx={{ gap: 2 }}>
              <HelpOutlineIcon fontSize="small" /> Help & Support
            </MenuItem>
          </Menu>
          
          {/* Notifications menu */}
          <Menu
            anchorEl={notifAnchorEl}
            open={Boolean(notifAnchorEl)}
            onClose={handleNotifClose}
            PaperProps={{
              elevation: 3,
              sx: { 
                mt: 1.5, 
                borderRadius: 2,
                minWidth: 280,
                maxWidth: 320,
                overflow: 'visible',
                '&:before': {
                  content: '""',
                  display: 'block',
                  position: 'absolute',
                  top: 0,
                  right: 14,
                  width: 10,
                  height: 10,
                  bgcolor: 'background.paper',
                  transform: 'translateY(-50%) rotate(45deg)',
                  zIndex: 0,
                },
              },
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <Box sx={{ p: 2, pb: 1 }}>
              <Typography variant="subtitle1" fontWeight={600}>
                Notifications
              </Typography>
              <Typography variant="body2" color="text.secondary">
                You have {mockNotifications.filter(n => !n.read).length} unread notifications
              </Typography>
            </Box>
            <Divider />
            {mockNotifications.map(notification => (
              <MenuItem key={notification.id} sx={{ px: 2, py: 1.5 }}>
                <Box sx={{ width: '100%' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="subtitle2">
                      {notification.title}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {notification.time}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    {notification.description}
                  </Typography>
                </Box>
              </MenuItem>
            ))}
            <Divider />
            <Box sx={{ p: 1.5, textAlign: 'center' }}>
              <Button size="small" onClick={() => showSnackbarMessage('Viewed all notifications', 'success')}>
                View All
              </Button>
            </Box>
          </Menu>
          
          {/* Drawer */}
          <Box
            component="nav"
            sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
          >
            {/* Mobile drawer - temporary */}
            {isMobile && (
              <Drawer
                variant="temporary"
                open={drawerOpen}
                onClose={handleDrawerToggle}
                ModalProps={{ keepMounted: true }}
                sx={{
                  '& .MuiDrawer-paper': { 
                    boxSizing: 'border-box', 
                    width: drawerWidth,
                    borderRight: 'none',
                    borderRadius: 0,
                  },
                  '& .MuiPaper-root': {
                    width: drawerWidth,
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  }
                }}
              >
                {drawer}
              </Drawer>
            )}
            
            {/* Desktop drawer - persistent */}
            {!isMobile && (
              <Drawer
                variant="persistent"
                open={drawerOpen}
                sx={{
                  '& .MuiDrawer-paper': { 
                    boxSizing: 'border-box', 
                    width: drawerWidth,
                    borderRight: `1px solid ${theme.palette.divider}`,
                    borderRadius: 0,
                    bgcolor: theme.palette.background.paper,
                    boxShadow: theme.palette.mode === 'light' 
                      ? '1px 0 5px 0 rgba(0,0,0,0.05)' 
                      : 'none',
                  },
                }}
              >
                {drawer}
              </Drawer>
            )}
          </Box>
          
          {/* Main Content */}
          <Main open={drawerOpen && !isMobile}>
            <Toolbar /> {/* For spacing below app bar */}
            <Container
              maxWidth={false}
              disableGutters
              sx={{ 
                pt: 0,
                pb: 4,
                px: 0,
                height: 'calc(100vh - 64px)',
                overflow: 'auto',
              }}
            >
              <Fade in={true} timeout={450}>
                <Box sx={{ maxWidth: '100%', overflowX: 'hidden', px: 0 }}>
                  <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/incidents" element={<CaseList />} />
                    <Route path="/incidents/:caseId" element={<CaseDetail />} />
                    <Route path="/incidents/:caseId/vehicles/:personId" element={<PersonDetail />} />
                    <Route path="/incidents/:caseId/timeline" element={<TimelineView />} />
                    
                    {/* Unified AI Assistant - works for both general questions and case analysis */}
                    <Route path="/assistant" element={<UnifiedChat />} />
                    <Route path="/incidents/:caseId/chat" element={<UnifiedChat />} />
                    
                    {/* Legacy routes (redirected to unified chat) */}
                    <Route path="/incident/:caseId/chat" element={<UnifiedChat />} />
                    <Route path="/legal-assistant" element={<UnifiedChat />} />
                    <Route path="/chat" element={<UnifiedChat />} />
                    
                    <Route path="/upload" element={<UploadForm />} />
                    <Route path="*" element={<Dashboard />} />
                    <Route path="/incidents/:caseId/video" element={<VideoPlayer />} />
                  </Routes>
                </Box>
              </Fade>
            </Container>
          </Main>
          
          <Snackbar 
            open={snackbar.open} 
            autoHideDuration={4000} 
            onClose={handleSnackbarClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            sx={{ mb: 2 }}
          >
            <Alert 
              onClose={handleSnackbarClose} 
              severity={snackbar.severity} 
              variant="filled"
              sx={{ width: '100%', boxShadow: '0 3px 10px rgba(0,0,0,0.2)' }}
            >
              {snackbar.message}
            </Alert>
          </Snackbar>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
               