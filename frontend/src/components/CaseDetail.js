import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Typography, Box, Grid, Paper, Tabs, Tab, Button, CircularProgress, Chip, Divider, Tooltip, Snackbar, Alert, Dialog, DialogTitle, DialogContent, DialogActions, TextField, IconButton } from '@mui/material';
import axios from 'axios';
import VideocamIcon from '@mui/icons-material/Videocam';
import DownloadIcon from '@mui/icons-material/Download';
import TimelineIcon from '@mui/icons-material/Timeline';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import MapIcon from '@mui/icons-material/Map';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';

function CaseDetail() {
  const navigate = useNavigate();
  const { caseId } = useParams();
  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [downloadingVideo, setDownloadingVideo] = useState(false);
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('info');
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [newCaseName, setNewCaseName] = useState('');
  const [renamingCase, setRenamingCase] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Current user and timestamp for logging
  const currentUser = "aaravgoel0";
  const currentDate = "2025-04-06 09:01:50";

  const formatNumber = (value, digits = 2) => {
    if (value === null || value === undefined) {
      return '—';
    }
    const numeric = typeof value === 'number' ? value : Number(value);
    if (Number.isNaN(numeric)) {
      return value;
    }
    if (digits === 0) {
      return Math.round(numeric).toLocaleString();
    }
    return numeric.toFixed(digits);
  };

  const formatPercent = (value, digits = 1) => {
    if (value === null || value === undefined) {
      return '—';
    }
    return `${formatNumber(value * 100, digits)}%`;
  };

  useEffect(() => {
    async function fetchCaseData() {
      try {
        setLoading(true);
        
        // Get case details
        const response = await axios.get(`/api/cases/${caseId}`);
        setCaseData(response.data);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching case details:', err);
        setError('Failed to load case data');
        setLoading(false);
      }
    }
    
    fetchCaseData();
  }, [caseId]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleDownloadVideo = async () => {
    if (!caseData?.output_video) return;
    
    try {
      setDownloadingVideo(true);
      
      // Use the dedicated download endpoint
      const downloadUrl = `/api/cases/${caseId}/download_video?user=aaravgoel0&t=${Date.now()}`;
      
      // Use fetch API to handle the download as a blob
      const response = await fetch(downloadUrl);
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      // Get the filename from the Content-Disposition header if available
      let filename = `analyzed_video_${caseId}.mp4`;
      const contentDisposition = response.headers.get('Content-Disposition');
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=(.+)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/["']/g, '');
        }
      }
      
      // Convert response to blob
      const blob = await response.blob();
      
      // Create object URL for the blob
      const url = window.URL.createObjectURL(blob);
      
      // Create download link
      const link = document.createElement('a');
      link.href = url;
      link.download = filename; // Set download attribute to force download
      
      // Append to document, click, and remove
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the object URL
      window.URL.revokeObjectURL(url);
      
      console.log(`Video download completed for case: ${caseId}`);
      
    } catch (err) {
      console.error('Error downloading video:', err);
      alert('Failed to download video. Please try again later.');
    } finally {
      setTimeout(() => {
        setDownloadingVideo(false);
      }, 1000);
    }
  };

  const handleCloseSnackbar = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setShowSnackbar(false);
  };

  const handleOpenRenameDialog = () => {
    setNewCaseName(caseData?.case_name || '');
    setRenameDialogOpen(true);
  };

  const handleCloseRenameDialog = () => {
    setRenameDialogOpen(false);
    setNewCaseName('');
  };

  const handleRenameCase = async () => {
    if (!newCaseName.trim()) {
      setSnackbarMessage('Case name cannot be empty');
      setSnackbarSeverity('error');
      setShowSnackbar(true);
      return;
    }

    try {
      setRenamingCase(true);
      const response = await axios.put(`/api/cases/${caseId}/rename`, {
        case_name: newCaseName.trim()
      });

      if (response.data.success) {
        // Update the local case data
        setCaseData(prev => ({
          ...prev,
          case_name: newCaseName.trim()
        }));

        setSnackbarMessage('Case renamed successfully');
        setSnackbarSeverity('success');
        setShowSnackbar(true);
        handleCloseRenameDialog();
      }
    } catch (err) {
      console.error('Error renaming case:', err);
      setSnackbarMessage('Failed to rename case. Please try again.');
      setSnackbarSeverity('error');
      setShowSnackbar(true);
    } finally {
      setRenamingCase(false);
    }
  };

  const handleDeleteClick = () => {
    setDeleteDialogOpen(true);
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
  };

  const handleDeleteConfirm = async () => {
    try {
      setDeleting(true);
      await axios.delete(`/api/cases/${caseId}`);
      
      setSnackbarMessage(`Case "${caseData.case_name || caseId}" deleted successfully`);
      setSnackbarSeverity('success');
      setShowSnackbar(true);
      
      // Navigate back to cases list after a short delay
      setTimeout(() => {
        navigate('/incidents');
      }, 1500);
    } catch (err) {
      console.error('Error deleting case:', err);
      setSnackbarMessage(err.response?.data?.error || 'Failed to delete case');
      setSnackbarSeverity('error');
      setShowSnackbar(true);
      setDeleting(false);
      setDeleteDialogOpen(false);
    }
  };

  if (loading || error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!caseData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <Typography>Case not found</Typography>
      </Box>
    );
  }

  const metrics = caseData.metrics || {};
  const drivingBehavior = caseData.driving_behavior || {};
  const vehicleSummary = drivingBehavior.vehicle_summary || {};
  const behaviorInsights = drivingBehavior.insights || [];
  const unsafeEvents = drivingBehavior.unsafe_events || [];
  const signInteractions = drivingBehavior.sign_interactions || [];
  const complianceSummary = drivingBehavior.compliance_summary || {};
  const vehicleCounts = caseData.vehicle_counts || {};
  const totalVehicleDetections =
    caseData.total_vehicle_detections ??
    caseData.num_vehicles ??
    Object.values(vehicleCounts).reduce((sum, count) => sum + (Number(count) || 0), 0);
  const vehicleTrackCount =
    vehicleSummary.total_vehicle_tracks ??
    metrics.vehicle_track_count ??
    (Array.isArray(caseData.vehicle_tracks) ? caseData.vehicle_tracks.length : 0);

  const normalizedVehicleClasses = new Set([
    ...Object.keys(vehicleCounts).map((name) => name.toLowerCase()),
    'car',
    'truck',
    'bus',
    'motorcycle',
    'bicycle',
    'van',
    'suv',
    'pickup',
    'vehicle',
  ]);

  const rawTrackList = Array.isArray(caseData.vehicle_tracks)
    ? caseData.vehicle_tracks
    : Array.isArray(caseData.tracks)
      ? caseData.tracks
      : caseData.track_details && typeof caseData.track_details === 'object'
        ? Object.values(caseData.track_details)
        : caseData.tracks && typeof caseData.tracks === 'object'
          ? Object.values(caseData.tracks)
          : [];

  const trackList = rawTrackList.filter((track) =>
    normalizedVehicleClasses.has((track?.class_name || '').toLowerCase())
  );

  const metricCards = [
    { label: 'Frames Processed', value: metrics.frames_processed, digits: 0 },
    { label: 'Avg Detections / Frame', value: metrics.average_detections_per_frame, digits: 2 },
    { label: 'Avg Raw Detections / Frame', value: metrics.average_raw_detections_per_frame, digits: 2 },
    { label: 'Max Detections In Frame', value: metrics.max_detections_in_frame, digits: 0 },
    { label: 'Avg Confidence', value: metrics.average_confidence, digits: 3 },
    { label: 'Filtered Ratio', value: metrics.filtered_ratio, format: (val) => formatPercent(val) },
    { label: 'Avg Inference (ms)', value: metrics.average_inference_ms, digits: 1 },
    { label: 'Processing FPS', value: metrics.processing_fps, digits: 1 },
    { label: 'Analysis Duration (s)', value: metrics.analysis_duration_sec, digits: 1 },
    { label: 'Vehicle Tracks', value: vehicleTrackCount, digits: 0 },
    { label: 'Vehicle Detections', value: totalVehicleDetections, digits: 0 },
    { label: 'High Speed Events', value: vehicleSummary.high_speed_events, digits: 0 },
    { label: 'Collision Flags', value: vehicleSummary.possible_collisions, digits: 0 },
  ];

  const drivingSummaryCards = [
    { label: 'Avg Track Duration (s)', value: vehicleSummary.average_track_duration_sec, digits: 1 },
    { label: 'Median Track Duration (s)', value: vehicleSummary.median_track_duration_sec, digits: 1 },
    { label: 'Average Speed (norm)', value: vehicleSummary.average_speed_norm, digits: 3 },
    { label: 'Max Speed (norm)', value: vehicleSummary.max_speed_norm, digits: 3 },
    { label: 'Lane Shift Events', value: vehicleSummary.lane_shift_events, digits: 0 },
  ].filter(({ value }) => value !== null && value !== undefined);

  const visibleMetricCards = metricCards.filter(({ value }) => value !== null && value !== undefined);

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h4">
            {caseData?.case_name || `Case: ${caseId}`}
          </Typography>
          <Tooltip title="Rename case">
            <IconButton 
              color="primary" 
              size="small"
              onClick={handleOpenRenameDialog}
              sx={{ 
                '&:hover': { 
                  backgroundColor: 'primary.light',
                  color: 'white'
                } 
              }}
            >
              <EditIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete case">
            <IconButton 
              color="error" 
              size="small"
              onClick={handleDeleteClick}
              sx={{ 
                '&:hover': { 
                  backgroundColor: 'error.light',
                  color: 'white'
                } 
              }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
        <Box>
          <Button 
            component={Link} 
            to="/cases" 
            variant="outlined" 
            color="primary" 
            sx={{ mr: 2 }}
          >
            Back to Cases
          </Button>
          <Button 
            component={Link} 
            to={`/cases/${caseId}/timeline`} 
            variant="contained" 
            color="primary" 
            startIcon={<TimelineIcon />}
          >
            Timeline View
          </Button>
        </Box>
      </Box>
      
      {/* Case Summary */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              {caseData.video_path?.split('/').pop() || 'Video Analysis'}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Analysis timestamp: {new Date(caseData.timestamp).toLocaleString()}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Total detections: {caseData.total_detections}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Unique objects: {caseData.total_unique_objects}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Vehicle tracks: {vehicleTrackCount ?? 0}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Vehicle detections: {totalVehicleDetections ?? 0}
            </Typography>
            
            {/* Display any 3D reconstruction data if available */}
            {caseData['3d_analysis'] && caseData['3d_analysis'].reconstruction_available && (
              <Box sx={{ mt: 2 }}>
                <Chip 
                  icon={<ViewInArIcon />} 
                  label="3D Reconstruction Available" 
                  color="success" 
                  variant="outlined" 
                />
              </Box>
            )}
            
            {/* Video Player Button */}
            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<VideoLibraryIcon />}
                onClick={() => navigate(`/cases/${caseId}/video`)}
                fullWidth
                sx={{ 
                  py: 1.5,
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    transition: 'transform 0.3s'
                  }
                }}
              >
                View Video in Interactive Player
              </Button>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            {caseData.output_video && (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                {/* Download button */}
                <Tooltip title="Download the analyzed video to your device">
                  <Button 
                    variant="contained" 
                    color="secondary" 
                    startIcon={downloadingVideo ? <CircularProgress size={20} color="inherit" /> : <DownloadIcon />}
                    onClick={handleDownloadVideo}
                    disabled={downloadingVideo}
                    sx={{ 
                      mx: 1,
                      position: 'relative',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        transition: 'transform 0.3s'
                      }
                    }}
                  >
                    {downloadingVideo ? 'Downloading...' : 'Download Analyzed Video'}
                  </Button>
                </Tooltip>
                
                {/* View button with purple outline - REMOVED */}
                
                {caseData.visualizations?.trajectories && (
                  <Button 
                    variant="contained" 
                    color="primary" 
                    startIcon={<MapIcon />}
                    href={caseData.visualizations.trajectories}
                    target="_blank"
                    sx={{ mx: 1 }}
                  >
                    View Trajectories
                  </Button>
                )}
              </Box>
            )}
          </Grid>
        </Grid>
      </Paper>
    
        {visibleMetricCards.length > 0 && (
          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <Typography variant="h5" gutterBottom>Analysis Metrics</Typography>
            <Grid container spacing={2}>
              {visibleMetricCards.map(({ label, value, digits = 2, format }) => (
                <Grid item xs={12} sm={6} md={4} lg={3} key={label}>
                  <Paper elevation={1} sx={{ p: 2, textAlign: 'center', height: '100%' }}>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {format ? format(value) : formatNumber(value, digits)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {label}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Paper>
        )}
      
      {/* Visualizations */}
      {caseData.visualizations && Object.keys(caseData.visualizations).length > 0 && (
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>Visualizations</Typography>
          
          <Grid container spacing={3}>
            {caseData.visualizations.timeline && (
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle1">Event Timeline</Typography>
                </Box>
                <img 
                  src={caseData.visualizations.timeline} 
                  alt="Event Timeline" 
                  style={{ width: '100%', height: 'auto', maxHeight: '400px', objectFit: 'contain', border: '1px solid #ddd' }}
                />
              </Grid>
            )}
            
            {caseData.visualizations.heatmap && (
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle1">Movement Heatmap</Typography>
                </Box>
                <img 
                  src={caseData.visualizations.heatmap} 
                  alt="Movement Heatmap" 
                  style={{ width: '100%', height: 'auto', maxHeight: '400px', objectFit: 'contain', border: '1px solid #ddd' }}
                />
              </Grid>
            )}
          </Grid>
        </Paper>
      )}
      
      {/* Driving insights and detection summaries */}
      <Paper elevation={3} sx={{ p: 3 }}>
        <Box sx={{ width: '100%' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange} aria-label="case data tabs">
              <Tab label="Driving Insights" id="tab-0" aria-controls="tabpanel-0" />
              <Tab label="Object Detections" id="tab-1" aria-controls="tabpanel-1" />
            </Tabs>
          </Box>

          {/* Driving Insights Tab */}
          <TabPanel value={activeTab} index={0}>
            <Typography variant="h5" gutterBottom>Driving Insights</Typography>

            {drivingBehavior?.error && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                Driving behavior analysis unavailable: {drivingBehavior.error}
              </Alert>
            )}

            {complianceSummary?.risk_score !== undefined && (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Risk score: {formatNumber(complianceSummary.risk_score, 1)} / 100
              </Typography>
            )}

            {drivingSummaryCards.length > 0 && (
              <Grid container spacing={2} sx={{ mb: 3 }}>
                {drivingSummaryCards.map(({ label, value, digits = 2 }) => (
                  <Grid item xs={12} sm={6} md={4} key={label}>
                    <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {formatNumber(value, digits)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {label}
                      </Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            )}

            {behaviorInsights.length > 0 && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>Insights</Typography>
                <ul>
                  {behaviorInsights.map((insight, idx) => (
                    <li key={`insight-${idx}`}>{insight}</li>
                  ))}
                </ul>
              </Box>
            )}

            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>Unsafe Events</Typography>
              {unsafeEvents.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No unsafe driving events flagged in this analysis.
                </Typography>
              ) : (
                <Grid container spacing={2}>
                  {unsafeEvents.map((event, idx) => (
                    <Grid item xs={12} md={6} key={`event-${idx}`}>
                      <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                          {event.type?.replace(/_/g, ' ') || 'Event'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Severity: {event.severity || 'unknown'}
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          {event.description || 'Behavior requires review.'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          Frame: {event.frame ?? '—'} | Time: {event.timestamp !== undefined ? `${formatNumber(event.timestamp, 2)}s` : '—'}
                        </Typography>
                        {Array.isArray(event.track_ids) && event.track_ids.length > 0 && (
                          <Typography variant="body2" color="text.secondary">
                            Track IDs: {event.track_ids.join(', ')}
                          </Typography>
                        )}
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </Box>

            <Box>
              <Typography variant="h6" gutterBottom>Sign Interactions</Typography>
              {signInteractions.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No traffic sign interactions detected.
                </Typography>
              ) : (
                <Grid container spacing={2}>
                  {signInteractions.slice(0, 6).map((interaction, idx) => (
                    <Grid item xs={12} md={6} key={`sign-${idx}`}>
                      <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                          {interaction.sign_class || 'Sign'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Frame: {interaction.frame ?? '—'} | Time: {interaction.timestamp !== undefined ? `${formatNumber(interaction.timestamp, 2)}s` : '—'}
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          Nearby vehicles: {Array.isArray(interaction.vehicles_in_vicinity) ? interaction.vehicles_in_vicinity.length : 0}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </Box>
          </TabPanel>

          {/* Object Detections Tab */}
          <TabPanel value={activeTab} index={1}>
            <Typography variant="h5" gutterBottom>Object Detections</Typography>

            <Grid container spacing={3}>
              {Object.entries(caseData.class_counts || {}).map(([className, count], index) => (
                <Grid item key={index} xs={6} sm={4} md={3} lg={2}>
                  <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h6">{count}</Typography>
                    <Typography variant="body2" color="text.secondary">{className}</Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>

            {trackList.length > 0 && (
              <Box sx={{ mt: 4 }}>
                <Typography variant="h6" gutterBottom>Vehicle Tracks</Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Showing top {Math.min(trackList.length, 10)} tracks by duration
                </Typography>

                <Grid container spacing={2}>
                  {trackList.slice(0, 10).map((track, index) => (
                    <Grid item key={index} xs={12} sm={6} md={4}>
                      <Paper elevation={2} sx={{ p: 2 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                          {(track.class_name || 'Vehicle')} #{track.id ?? track.object_id ?? index + 1}
                        </Typography>
                        <Typography variant="body2">
                          Duration: {track.duration !== undefined ? `${formatNumber(track.duration, 2)} seconds` : 'unknown'}
                        </Typography>
                        <Typography variant="body2">
                          Frames: {track.first_frame ?? '—'} - {track.last_frame ?? '—'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Detections: {track.num_detections ?? track.detections?.length ?? '—'}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}
          </TabPanel>
        </Box>
      </Paper>
      
      {/* Snackbar for notifications */}
      <Snackbar 
        open={showSnackbar} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>

      {/* Rename Dialog */}
      <Dialog 
        open={renameDialogOpen} 
        onClose={handleCloseRenameDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Rename Case</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Case Name"
            type="text"
            fullWidth
            variant="outlined"
            value={newCaseName}
            onChange={(e) => setNewCaseName(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !renamingCase) {
                handleRenameCase();
              }
            }}
            disabled={renamingCase}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseRenameDialog} disabled={renamingCase}>
            Cancel
          </Button>
          <Button 
            onClick={handleRenameCase} 
            variant="contained" 
            disabled={renamingCase || !newCaseName.trim()}
          >
            {renamingCase ? <CircularProgress size={24} /> : 'Rename'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
      >
        <DialogTitle>Delete Case</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete case "{caseData?.case_name || caseId}"? 
            This action cannot be undone and will permanently delete all associated data, videos, and analysis results.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel} disabled={deleting}>
            Cancel
          </Button>
          <Button 
            onClick={handleDeleteConfirm} 
            color="error" 
            variant="contained"
            disabled={deleting}
            startIcon={deleting ? <CircularProgress size={20} /> : <DeleteIcon />}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// Tab Panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
      style={{ padding: '20px 0' }}
    >
      {value === index && (
        <Box>
          {children}
        </Box>
      )}
    </div>
  );
}

export default CaseDetail;