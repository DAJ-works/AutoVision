import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Button, Grid, Paper, Fade, Grow, 
  Container, Card, CardContent, Avatar, 
  useTheme
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Link } from 'react-router-dom';

// Icons
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import VideocamIcon from '@mui/icons-material/Videocam';
import DirectionsCarIcon from '@mui/icons-material/DirectionsCar';
import TimelineIcon from '@mui/icons-material/Timeline';
import BarChartIcon from '@mui/icons-material/BarChart';
import SpeedIcon from '@mui/icons-material/Speed';
import TrafficIcon from '@mui/icons-material/Traffic';
import GavelIcon from '@mui/icons-material/Gavel';
import AssignmentIcon from '@mui/icons-material/Assignment';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import LocalShippingIcon from '@mui/icons-material/LocalShipping';
import WarningIcon from '@mui/icons-material/Warning';
import FolderIcon from '@mui/icons-material/Folder';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import MapIcon from '@mui/icons-material/Map';

// Styled Components
const HeroSection = styled(Box)(({ theme }) => ({
  position: 'relative',
  minHeight: '600px',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  overflow: 'hidden',
  padding: theme.spacing(8, 0),
  background: theme.palette.mode === 'dark' 
    ? 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)'
    : 'linear-gradient(135deg, #f8fafc 0%, #e0e7ff 50%, #f8fafc 100%)',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: '-50%',
    right: '-20%',
    width: '800px',
    height: '800px',
    background: 'radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%)',
    borderRadius: '50%',
    animation: 'float 20s ease-in-out infinite',
  },
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: '-30%',
    left: '-10%',
    width: '600px',
    height: '600px',
    background: 'radial-gradient(circle, rgba(139, 92, 246, 0.12) 0%, transparent 70%)',
    borderRadius: '50%',
    animation: 'float 15s ease-in-out infinite reverse',
  },
  '@keyframes float': {
    '0%, 100%': { transform: 'translate(0, 0) scale(1)' },
    '33%': { transform: 'translate(30px, -50px) scale(1.1)' },
    '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
  },
  [theme.breakpoints.up('md')]: {
    padding: theme.spacing(12, 6),
    minHeight: '700px',
  },
}));

const FeatureCard = styled(Card)(({ theme }) => ({
  height: '100%',
  borderRadius: 20,
  transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
  boxShadow: theme.palette.mode === 'dark'
    ? '0 4px 20px rgba(0,0,0,0.3)'
    : '0 4px 20px rgba(0,0,0,0.08)',
  overflow: 'hidden',
  border: '1px solid',
  borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)',
  background: theme.palette.mode === 'dark'
    ? 'linear-gradient(145deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%)'
    : 'linear-gradient(145deg, rgba(255, 255, 255, 0.98) 0%, rgba(249, 250, 251, 1) 100%)',
  backdropFilter: 'blur(20px)',
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '4px',
    background: 'linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%)',
    opacity: 0,
    transition: 'opacity 0.4s',
  },
  '&:hover': {
    transform: 'translateY(-12px) scale(1.02)',
    boxShadow: theme.palette.mode === 'dark'
      ? '0 20px 40px rgba(59, 130, 246, 0.3)'
      : '0 16px 40px rgba(59, 130, 246, 0.2)',
    borderColor: theme.palette.primary.main,
    '&::before': {
      opacity: 1,
    }
  },
}));

const StatCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: 8,
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  alignItems: 'center',
  boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
}));

const FeatureIcon = styled(Avatar)(({ theme }) => ({
  width: 64,
  height: 64,
  backgroundColor: theme.palette.primary.main,
  boxShadow: '0 8px 24px rgba(59, 130, 246, 0.4)',
  marginBottom: theme.spacing(2),
  transition: 'all 0.3s ease',
  background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
  '.MuiCard-root:hover &': {
    transform: 'scale(1.15) rotate(8deg)',
    boxShadow: '0 12px 32px rgba(59, 130, 246, 0.5)',
  }
}));

// Badge styling
const Badge = styled(Box)(({ theme }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  padding: '8px 20px',
  borderRadius: 50,
  background: theme.palette.mode === 'dark'
    ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)'
    : 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
  border: '1px solid',
  borderColor: theme.palette.mode === 'dark' ? 'rgba(59, 130, 246, 0.3)' : 'rgba(59, 130, 246, 0.2)',
  marginBottom: theme.spacing(4),
  fontWeight: 700,
  fontSize: '0.9rem',
  boxShadow: '0 4px 16px rgba(59, 130, 246, 0.2)',
  position: 'relative',
  zIndex: 1,
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 700,
  letterSpacing: '-0.02em',
  marginBottom: theme.spacing(1),
}));

const HomePage = () => {
  const theme = useTheme();
  const [animateHero, setAnimateHero] = useState(false);
  const [animateFeatures, setAnimateFeatures] = useState(false);

  useEffect(() => {
    // Trigger animations after component mount
    setAnimateHero(true);
    
    // Stagger the animations
    const timer = setTimeout(() => {
      setAnimateFeatures(true);
    }, 400);
    
    return () => clearTimeout(timer);
  }, []);

  // Statistics for the stats section
  const stats = [
    { label: 'Analysis Accuracy', icon: <VerifiedUserIcon fontSize="large" color="primary" /> },
    { label: 'Incidents Analyzed', icon: <TrafficIcon fontSize="large" color="primary" /> },
    { label: 'Vehicles Tracked', icon: <AssignmentIcon fontSize="large" color="primary" /> },
    { label: 'City Partners', icon: <LocalShippingIcon fontSize="large" color="primary" /> },
  ];

  // Feature data with traffic analysis focus
  const features = [
    {
      title: 'Intelligent Vehicle Tracking',
      description: 'Advanced AI algorithms detect and track every vehicle across frames with unique IDs and trajectory analysis.',
      icon: <DirectionsCarIcon fontSize="large" />,
      color: theme.palette.primary.main,
      link: '/cases'
    },
    {
      title: 'Speed & Trajectory Estimation',
      description: 'Optical flow and perspective mapping calculate vehicle speeds and predict collision paths in real-time.',
      icon: <SpeedIcon fontSize="large" />,
      color: theme.palette.primary.dark,
      link: '/cases'
    },
    {
      title: 'Accident Reconstruction',
      description: 'Frame-by-frame collision analysis with physics-based modeling to determine exact sequence of events.',
      icon: <TimelineIcon fontSize="large" />,
      color: theme.palette.primary.main,
      link: '/cases'
    },
    {
      title: 'Violation Detection',
      description: 'Automatically identifies red-light violations, speeding, unsafe lane changes, and illegal maneuvers.',
      icon: <WarningIcon fontSize="large" />,
      color: theme.palette.error.main,
      link: '/cases'
    },
    {
      title: 'Fault Determination AI',
      description: 'Applies traffic laws and physics to determine which driver caused the incident with detailed explanations.',
      icon: <GavelIcon fontSize="large" />,
      color: theme.palette.primary.dark,
      link: '/chat'
    },
    {
      title: 'Traffic Flow Analysis',
      description: 'Identify dangerous intersections and patterns to improve road safety and urban planning.',
      icon: <MapIcon fontSize="large" />,
      color: theme.palette.info.main,
      link: '/cases'
    },
  ];

  return (
    <Box sx={{ overflowX: 'hidden', mx: 0 }}>
      {/* Hero Section */}
      <HeroSection>
        <Container maxWidth={false} disableGutters>
          <Fade in={animateHero} timeout={1000}>
            <Box>
              <Badge>
                <TrafficIcon fontSize="small" sx={{ mr: 1 }} />
                AutoVision Platform • Intelligent Traffic Analysis
              </Badge>
              
              <Typography 
                variant="h1" 
                component="h1" 
                sx={{ 
                  fontWeight: 900, 
                  fontSize: { xs: '2.5rem', sm: '3rem', md: '4rem' },
                  lineHeight: 1.1,
                  mb: 3,
                  letterSpacing: '-0.03em',
                  background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%)',
                  backgroundSize: '200% 200%',
                  animation: 'gradient-shift 8s ease infinite',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  position: 'relative',
                  zIndex: 1,
                  '@keyframes gradient-shift': {
                    '0%': { backgroundPosition: '0% 50%' },
                    '50%': { backgroundPosition: '100% 50%' },
                    '100%': { backgroundPosition: '0% 50%' },
                  }
                }}
              >
                AI-Powered Accident Reconstruction<br/>
                & Traffic Safety Analysis
              </Typography>
              
              <Typography 
                variant="h5" 
                color="text.secondary" 
                sx={{ 
                  maxWidth: 700, 
                  mb: 5, 
                  fontWeight: 500,
                  lineHeight: 1.6,
                  position: 'relative',
                  zIndex: 1,
                  fontSize: { xs: '1.1rem', md: '1.3rem' }
                }}
              >
                Automatically analyze dashcam and traffic footage to reconstruct accidents, determine fault, and improve road safety with intelligent AI-driven insights.
              </Typography>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, position: 'relative', zIndex: 1 }}>
                <Button
                  component={Link}
                  to="/dashboard"
                  variant="contained"
                  size="large"
                  startIcon={<BarChartIcon />}
                  sx={{
                    px: 4,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    borderRadius: 3,
                    background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                    boxShadow: '0 8px 32px rgba(59, 130, 246, 0.4)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-4px) scale(1.05)',
                      boxShadow: '0 12px 40px rgba(59, 130, 246, 0.5)',
                      background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                    }
                  }}
                >
                  Get Started
                </Button>
                <Button
                  component={Link}
                  to="/upload"
                  variant="outlined"
                  size="large"
                  startIcon={<VideocamIcon />}
                  sx={{
                    px: 4,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    borderRadius: 3,
                    borderWidth: 2,
                    borderColor: 'primary.main',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      borderWidth: 2,
                      transform: 'translateY(-4px)',
                      background: 'rgba(59, 130, 246, 0.1)',
                      boxShadow: '0 8px 24px rgba(59, 130, 246, 0.2)',
                    }
                  }}
                >
                  Upload Video
                </Button>
              </Box>
            </Box>
          </Fade>
        </Container>
        
        {/* Subtle decorative element */}
        <Box sx={{ 
          position: 'absolute', 
          width: '100%',
          height: '100%',
          top: 0,
          left: 0,
          opacity: 0.03,
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000...`,
          zIndex: 0
        }} />
      </HeroSection>

      {/* Main Features Section */}
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography
            variant="overline"
            sx={{
              color: 'primary.main',
              fontWeight: 700,
              fontSize: '0.9rem',
              letterSpacing: 2,
              mb: 2,
              display: 'block'
            }}
          >
            POWERFUL FEATURES
          </Typography>
          <SectionTitle 
            variant="h3"
            sx={{
              fontWeight: 900,
              fontSize: { xs: '2rem', md: '2.5rem' },
              letterSpacing: '-0.02em',
              mb: 2,
              background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Intelligent Traffic Analysis Tools
          </SectionTitle>
          <Typography 
            variant="h6" 
            color="text.secondary" 
            sx={{ 
              maxWidth: 800,
              mx: 'auto',
              fontWeight: 400,
              lineHeight: 1.6
            }}
          >
            Our platform provides automated accident reconstruction, fault determination, and traffic safety insights using advanced AI and computer vision.
          </Typography>
        </Box>
        
        <Grid container spacing={3}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Grow in={animateFeatures} timeout={300 + index * 100}>
                <Box>
                  <FeatureCard>
                    <CardContent sx={{ p: 3 }}>
                      <FeatureIcon sx={{ bgcolor: feature.color }}>
                        {feature.icon}
                      </FeatureIcon>
                      
                      <Typography 
                        variant="h6" 
                        component="h3"
                        sx={{ 
                          fontWeight: 600, 
                          mb: 1.5,
                        }}
                      >
                        {feature.title}
                      </Typography>
                      
                      <Typography 
                        variant="body2" 
                        color="text.secondary"
                        sx={{ 
                          mb: 2,
                          lineHeight: 1.6
                        }}
                      >
                        {feature.description}
                      </Typography>
                      
                      <Button 
                        component={Link}
                        to={feature.link}
                        endIcon={<ArrowForwardIcon />}
                        size="small"
                        sx={{ 
                          textTransform: 'none',
                          fontWeight: 500
                        }}
                      >
                        Access module
                      </Button>
                    </CardContent>
                  </FeatureCard>
                </Box>
              </Grow>
            </Grid>
          ))}
        </Grid>
      </Container>
      
      {/* Quick Access Section */}
      <Container maxWidth="lg" sx={{ pb: 8, pt: 2 }}>
        <SectionTitle variant="h5" sx={{ mb: 3 }}>
          Quick Access
        </SectionTitle>
        
        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <Fade in={animateFeatures} timeout={600}>
              <Paper 
                component={Link} 
                to="/dashboard"
                elevation={0} 
                sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  textDecoration: 'none',
                  color: 'text.primary',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: theme.palette.primary.main, 
                    width: 40, 
                    height: 40,
                  }}
                >
                  <BarChartIcon />
                </Avatar>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  Dashboard
                </Typography>
              </Paper>
            </Fade>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Fade in={animateFeatures} timeout={800}>
              <Paper 
                component={Link} 
                to="/incidents"
                elevation={0} 
                sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  textDecoration: 'none',
                  color: 'text.primary',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: theme.palette.primary.main, 
                    width: 40, 
                    height: 40,
                  }}
                >
                  <FolderIcon />
                </Avatar>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  Incidents
                </Typography>
              </Paper>
            </Fade>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Fade in={animateFeatures} timeout={900}>
              <Paper 
                component={Link} 
                to="/assistant"
                elevation={0} 
                sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  textDecoration: 'none',
                  color: 'text.primary',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: theme.palette.primary.main, 
                    width: 40, 
                    height: 40,
                  }}
                >
                  <GavelIcon />
                </Avatar>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  AI Assistant
                </Typography>
              </Paper>
            </Fade>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Fade in={animateFeatures} timeout={1000}>
              <Paper 
                component={Link} 
                to="/upload"
                elevation={0} 
                sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  textDecoration: 'none',
                  color: 'text.primary',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: theme.palette.primary.main, 
                    width: 40, 
                    height: 40,
                  }}
                >
                  <VideocamIcon />
                </Avatar>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  Upload
                </Typography>
              </Paper>
            </Fade>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default HomePage;