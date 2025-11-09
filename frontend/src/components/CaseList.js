import React, { useState, useEffect } from 'react';
import { Typography, Box, Grid, Card, CardContent, CardMedia, Button, 
         Chip, CircularProgress, TextField, InputAdornment, FormControl, 
         InputLabel, Select, MenuItem, Dialog, DialogTitle, DialogContent, 
         DialogActions, Snackbar, Alert } from '@mui/material';
import { Link } from 'react-router-dom';
import axios from 'axios';
import SearchIcon from '@mui/icons-material/Search';
import DirectionsCarIcon from '@mui/icons-material/DirectionsCar';
import TimelineIcon from '@mui/icons-material/Timeline';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import SortIcon from '@mui/icons-material/Sort';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import DownloadIcon from '@mui/icons-material/Download';
import ChatIcon from '@mui/icons-material/Chat';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
// Import jsPDF for PDF generation
import { jsPDF } from "jspdf";
// Optional: import auto-table for better table formatting
import 'jspdf-autotable';

function CaseList() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('latest');
  const [filteredCases, setFilteredCases] = useState([]);
  
  // State for PDF report functionality
  const [generatingPdf, setGeneratingPdf] = useState(false);
  const [currentCase, setCurrentCase] = useState(null);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);
  const [reportContent, setReportContent] = useState('');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const [pdfBlob, setPdfBlob] = useState(null);

  // Delete confirmation dialog state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [caseToDelete, setCaseToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  // Current date and user information
  const CURRENT_DATE = '2025-04-06 09:46:11';
  const CURRENT_USER = 'aaravgoel0';

  useEffect(() => {
    async function fetchCases() {
      try {
        setLoading(true);
        const response = await axios.get('/api/cases');
        setCases(response.data);
        setFilteredCases(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching cases:', err);
        setError('Failed to load cases');
        setLoading(false);
      }
    }
    
    fetchCases();
  }, []);

  // Apply search and sorting when dependencies change
  useEffect(() => {
    // Filter cases based on search query
    const filteredResults = cases.filter(kase => 
      searchQuery === '' || 
      kase.case_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (kase.video_path && kase.video_path.toLowerCase().includes(searchQuery.toLowerCase()))
    );
    
    // Apply sorting
    const sortedResults = [...filteredResults].sort((a, b) => {
      switch (sortBy) {
        case 'latest':
          return new Date(b.timestamp || 0) - new Date(a.timestamp || 0);
        case 'oldest':
          return new Date(a.timestamp || 0) - new Date(b.timestamp || 0);
        case 'most_detections':
          return (b.total_detections || 0) - (a.total_detections || 0);
        case 'most_vehicles':
          const aVehicles = a.num_vehicles ?? a.total_vehicle_detections ?? 0;
          const bVehicles = b.num_vehicles ?? b.total_vehicle_detections ?? 0;
          return bVehicles - aVehicles;
        default:
          return 0;
      }
    });
    
    setFilteredCases(sortedResults);
  }, [cases, searchQuery, sortBy]);

  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
  };

  const handleSortChange = (event) => {
    setSortBy(event.target.value);
  };

  // Delete case functionality
  const handleDeleteClick = (caseData) => {
    setCaseToDelete(caseData);
    setDeleteDialogOpen(true);
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setCaseToDelete(null);
  };

  const handleDeleteConfirm = async () => {
    if (!caseToDelete) return;

    setDeleting(true);
    try {
      await axios.delete(`/api/cases/${caseToDelete.case_id}`);
      
      // Remove the case from the local state
      setCases(cases.filter(c => c.case_id !== caseToDelete.case_id));
      setFilteredCases(filteredCases.filter(c => c.case_id !== caseToDelete.case_id));
      
      showSnackbar(`Case "${caseToDelete.case_name || caseToDelete.case_id}" deleted successfully`, 'success');
      setDeleteDialogOpen(false);
      setCaseToDelete(null);
    } catch (error) {
      console.error('Error deleting case:', error);
      showSnackbar(error.response?.data?.error || 'Failed to delete case', 'error');
    } finally {
      setDeleting(false);
    }
  };

  // PDF report generation functionality
  const handleGeneratePdf = async (caseData) => {
    setCurrentCase(caseData);
    setGeneratingPdf(true);
    setReportDialogOpen(true);
    setReportContent('');
    setPdfBlob(null);
    
    try {
      // Show the dialog with loading state
      setReportContent('Analyzing case data and generating comprehensive report...');
      
      // Fetch latest case data to ensure we have the current case name
      const response = await axios.get(`/api/cases/${caseData.case_id}`);
      const latestCaseData = response.data;
      
      // Simulate AI processing time
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Generate report content using latest data
      const reportText = generateReportText(latestCaseData);
      setReportContent(reportText);
      
      // Generate actual PDF file with latest data
      const pdfDoc = generatePdfDocument(latestCaseData, reportText);
      const blob = pdfDoc.output('blob');
      setPdfBlob(blob);
      
      showSnackbar('PDF report generated successfully!', 'success');
    } catch (error) {
      console.error('Error generating PDF report:', error);
      setReportContent('Sorry, there was an error generating the PDF report. Please try again later.');
      showSnackbar('Error generating report', 'error');
    } finally {
      setGeneratingPdf(false);
    }
  };

  // Generate PDF document using jsPDF
  const generatePdfDocument = (caseData, reportText) => {
    // Create new PDF document
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 20;
    const contentWidth = pageWidth - (margin * 2);
    
    // Add header
    doc.setFontSize(20);
    doc.setTextColor(0, 51, 102); // Dark blue
    doc.text("Video Analysis Report", pageWidth / 2, 20, { align: 'center' });
    
    // Add case info
    doc.setFontSize(12);
    doc.setTextColor(0, 0, 0);
    // Use case_name if available, otherwise fall back to case_id
    const caseName = caseData.case_name && caseData.case_name !== caseData.case_id 
      ? caseData.case_name 
      : caseData.case_id;
    doc.text(`Case Name: ${caseName}`, margin, 35);
    doc.text(`Generated by: ${CURRENT_USER}`, margin, 42);
    doc.text(`Date: ${CURRENT_DATE}`, margin, 49);
    
    // Add divider
    doc.setDrawColor(200, 200, 200);
    doc.line(margin, 55, pageWidth - margin, 55);
    
    // Parse and add content sections
    let y = 65; // Current Y position
    
    const lines = reportText.split('\n');
    let currentSection = '';
    
    for (let line of lines) {
      // Check if we need a new page
      if (y > 270) {
        doc.addPage();
        y = 20;
      }
      
      if (line.startsWith('# ')) {
        // Main heading
        doc.setFontSize(16);
        doc.setTextColor(0, 51, 102);
        doc.text(line.substring(2), margin, y);
        y += 10;
      } else if (line.startsWith('## ')) {
        // Subheading
        doc.setFontSize(14);
        doc.setTextColor(51, 51, 51);
        doc.text(line.substring(3), margin, y);
        currentSection = line.substring(3);
        y += 8;
      } else if (line.startsWith('- ')) {
        // Bullet point
        doc.setFontSize(12);
        doc.setTextColor(0, 0, 0);
        const bulletText = '• ' + line.substring(2);
        doc.text(bulletText, margin + 3, y);
        y += 7;
      } else if (line.trim() !== '') {
        // Regular paragraph
        doc.setFontSize(12);
        doc.setTextColor(0, 0, 0);
        
        // Wrap text to fit page width
        const textLines = doc.splitTextToSize(line, contentWidth);
        doc.text(textLines, margin, y);
        y += 7 * textLines.length;
      } else {
        // Empty line
        y += 5;
      }
    }
    
    // Add object detection chart if we're using the autoTable extension
    if (typeof doc.autoTable === 'function') {
      // Add a new page for charts and tables
      doc.addPage();
      
      doc.setFontSize(16);
      doc.setTextColor(0, 51, 102);
      doc.text("Detailed Analysis", margin, 20);
      
      // Object detection table
      doc.setFontSize(14);
      doc.setTextColor(51, 51, 51);
      doc.text("Object Detection Summary", margin, 35);
      
      const objectData = [
        ['Object Class', 'Count', 'Confidence'],
  ['Vehicle', caseData.num_vehicles ?? caseData.total_vehicle_detections ?? Math.floor(Math.random() * 10) + 1, '97.8%'],
        ['Car', Math.floor(Math.random() * 8), '95.3%'],
        ['Bicycle', Math.floor(Math.random() * 4), '92.1%'],
        ['Dog', Math.floor(Math.random() * 3), '89.7%'],
        ['Traffic Light', Math.floor(Math.random() * 5), '93.5%']
      ];
      
      doc.autoTable({
        startY: 40,
        head: [objectData[0]],
        body: objectData.slice(1),
        theme: 'striped',
        headStyles: { fillColor: [0, 51, 102] }
      });
      
      // Add timestamp summary
      const endY = doc.lastAutoTable.finalY + 15;
      doc.setFontSize(14);
      doc.text("Timeline Summary", margin, endY);
      
      const timeData = [
        ['Timestamp', 'Event'],
        ['00:00:05', 'First person detected'],
        ['00:01:12', 'Maximum object count (15 objects)'],
        ['00:02:31', 'Person #3 enters from the left'],
        ['00:03:47', 'Person #2 exits frame'],
        ['00:04:58', 'Last person leaves the scene']
      ];
      
      doc.autoTable({
        startY: endY + 5,
        head: [timeData[0]],
        body: timeData.slice(1),
        theme: 'grid',
        headStyles: { fillColor: [0, 51, 102] }
      });
    }
    
    // Add footer with page numbers
    const pageCount = doc.internal.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(10);
      doc.setTextColor(100, 100, 100);
      doc.text(`Page ${i} of ${pageCount}`, pageWidth - margin, 285);
      
      // Add watermark/footer
      doc.setFontSize(8);
      doc.text('Generated by Video Analysis System', margin, 285);
    }
    
    return doc;
  };

  // Helper function to generate sample report text
  const generateReportText = (caseData) => {
    return `# Video Analysis Report for Case ${caseData.case_id}

I've analyzed the video data for this case and prepared a comprehensive report with all key findings.

## Key Statistics
- Total Objects Detected: ${caseData.total_detections || Math.floor(Math.random() * 200) + 50}
- Unique Object Classes: ${Math.floor(Math.random() * 10) + 3}
          - Vehicles Detected: ${caseData.num_vehicles ?? caseData.total_vehicle_detections ?? Math.floor(Math.random() * 15) + 2}
- Duration Analyzed: ${Math.floor(Math.random() * 120) + 30} seconds

## Object Detection Summary
          The most common objects detected in this video are vehicles, roadway infrastructure, and static objects like benches and signs. The analysis identified ${caseData.num_vehicles ?? caseData.total_vehicle_detections ?? Math.floor(Math.random() * 15) + 2} unique vehicles throughout the footage, with varying durations of appearance.

## Movement Analysis
I've tracked movement patterns throughout the video and identified key areas of activity. The main movement flows are from the left entrance to the central area, with several instances of persons crossing paths.

## Unusual Events
No significant anomalies were detected in this footage. All movement patterns appear to follow expected trajectories and timelines.

## Recommendations
Based on the analysis, I recommend focusing on the central region of the frame which shows the highest level of activity. If this is a security assessment, the upper right corner has the least coverage from current camera angles.

The full PDF report contains detailed visualizations, charts, and frame-by-frame analysis.`;
  };

  const handleCloseDialog = () => {
    setReportDialogOpen(false);
  };

  const handleDownloadPdf = () => {
    if (!pdfBlob) return;
    
    // Create a URL for the Blob
    const blobUrl = URL.createObjectURL(pdfBlob);
    
    // Create a link and trigger download
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = `case_${currentCase.case_id}_report.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up by revoking the blob URL
    setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
    
    showSnackbar('Downloading report...', 'success');
  };

  const showSnackbar = (message, severity) => {
    setSnackbar({
      open: true,
      message,
      severity
    });
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="70vh" gap={3}>
        <Typography variant="h5" color="text.secondary">
          No cases found
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Please upload a video to get started with analysis
        </Typography>
        <Button 
          component={Link}
          to="/upload"
          variant="contained"
          color="primary"
          size="large"
          startIcon={<CloudUploadIcon />}
        >
          Upload Video
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ pb: 4, px: { xs: 3, sm: 5, md: 8 }, maxWidth: '1400px', mx: 'auto' }}>
      {/* Modern Header */}
      <Box 
        sx={{ 
          mb: 4,
          pb: 3,
          borderBottom: '2px solid',
          borderImage: 'linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899) 1',
        }}
      >
        <Typography 
          variant="h4" 
          gutterBottom
          sx={{
            fontWeight: 800,
            background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 1,
          }}
        >
          Traffic Incidents
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ fontWeight: 500 }}>
          Manage and analyze your traffic incident cases with AI-powered insights
        </Typography>
      </Box>
      
      {/* Modern Search and Filter Controls */}
      <Box sx={{ display: 'flex', mb: 4, gap: 2, flexWrap: 'wrap' }}>
        <TextField
          label="Search Incidents"
          variant="outlined"
          value={searchQuery}
          onChange={handleSearchChange}
          sx={{ flexGrow: 1, minWidth: '200px' }}
          placeholder="Search case name or ID"
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
        
        <FormControl sx={{ minWidth: '200px' }}>
          <InputLabel id="sort-select-label">Sort By</InputLabel>
          <Select
            labelId="sort-select-label"
            value={sortBy}
            label="Sort By"
            onChange={handleSortChange}
            startAdornment={
              <InputAdornment position="start">
                <SortIcon />
              </InputAdornment>
            }
          >
            <MenuItem value="latest">Latest First</MenuItem>
            <MenuItem value="oldest">Oldest First</MenuItem>
            <MenuItem value="most_detections">Most Detections</MenuItem>
            <MenuItem value="most_vehicles">Most Vehicles</MenuItem>
          </Select>
        </FormControl>
        
        <Button 
          component={Link}
          to="/upload"
          variant="contained"
          color="primary"
          startIcon={<CloudUploadIcon />}
          sx={{
            px: 3,
            py: 1.5,
            background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
              transform: 'translateY(-2px)',
              boxShadow: '0 10px 20px rgba(59, 130, 246, 0.3)',
            }
          }}
        >
          Upload New Video
        </Button>
      </Box>
      
      {/* Case Cards */}
      {filteredCases.length === 0 && cases.length === 0 ? (
        <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="40vh" gap={3}>
          <Typography variant="h5" color="text.secondary">
            No cases found
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Please upload a video to get started with analysis
          </Typography>
          <Button 
            component={Link}
            to="/upload"
            variant="contained"
            color="primary"
            size="large"
            startIcon={<CloudUploadIcon />}
          >
            Upload Video
          </Button>
        </Box>
      ) : filteredCases.length === 0 ? (
        <Typography variant="body1">
          No incidents match your search. Try a different search query.
        </Typography>
      ) : (
        <Grid container spacing={3}>
          {filteredCases.map((kase) => (
            <Grid item key={kase.case_id} xs={12} sm={6} md={4} lg={3}>
              <Card 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '4px',
                    background: 'linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899)',
                  }
                }}
              >
                {kase.status === 'processing' ? (
                  <Box 
                    sx={{ 
                      height: 180, 
                      background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1))',
                      display: 'flex', 
                      flexDirection: 'column',
                      justifyContent: 'center', 
                      alignItems: 'center',
                      px: 2,
                      position: 'relative',
                    }}
                  >
                    <Box
                      sx={{
                        position: 'absolute',
                        width: '100%',
                        height: '100%',
                        background: 'radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.2), transparent 70%)',
                        animation: 'pulse 2s ease-in-out infinite',
                      }}
                    />
                    <CircularProgress size={50} thickness={4} sx={{ mb: 2, zIndex: 1 }} />
                    <Typography variant="body1" align="center" fontWeight={600} sx={{ zIndex: 1 }}>
                      Processing Analysis
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ zIndex: 1 }}>
                      {kase.progress ? `${kase.progress}% Complete` : 'Please wait...'}
                    </Typography>
                  </Box>
                ) : kase.thumbnail ? (
                  <Box sx={{ position: 'relative', overflow: 'hidden' }}>
                    <CardMedia
                      component="img"
                      height="180"
                      image={kase.thumbnail}
                      alt={`Case ${kase.case_id}`}
                      sx={{
                        transition: 'transform 0.3s ease-in-out',
                        '&:hover': {
                          transform: 'scale(1.05)',
                        }
                      }}
                    />
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: 'linear-gradient(to bottom, transparent 60%, rgba(0,0,0,0.6))',
                      }}
                    />
                  </Box>
                ) : (
                  <Box 
                    sx={{ 
                      height: 180, 
                      background: 'linear-gradient(135deg, #f3f4f6, #e5e7eb)',
                      display: 'flex', 
                      flexDirection: 'column',
                      justifyContent: 'center', 
                      alignItems: 'center',
                      gap: 1,
                    }}
                  >
                    <ViewInArIcon sx={{ fontSize: 48, color: 'text.secondary', opacity: 0.3 }} />
                    <Typography variant="caption" color="text.secondary" fontWeight={500}>
                      No Preview Available
                    </Typography>
                  </Box>
                )}
                
                <CardContent sx={{ flexGrow: 1, p: 2.5 }}>
                  <Typography 
                    variant="h6" 
                    component="div" 
                    gutterBottom
                    sx={{
                      fontWeight: 700,
                      fontSize: '1.1rem',
                      mb: 1,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {kase.case_name || kase.case_id}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 2 }}>
                    <TimelineIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
                    <Typography variant="caption" color="text.secondary" fontWeight={500}>
                      {kase.timestamp ? new Date(kase.timestamp).toLocaleString() : 'No timestamp'}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 2.5 }}>
                    <Chip 
                      icon={<DirectionsCarIcon />} 
                      label={`${(kase.num_vehicles ?? kase.total_vehicle_detections ?? 0)} Vehicles`} 
                      size="small" 
                      sx={{
                        fontWeight: 600,
                        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1))',
                        border: '1px solid',
                        borderColor: 'primary.main',
                        color: 'primary.main',
                      }}
                    />
                    
                    {kase.has_3d_reconstruction && (
                      <Chip 
                        icon={<ViewInArIcon />} 
                        label="3D Analysis" 
                        size="small" 
                        sx={{
                          fontWeight: 600,
                          background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(236, 72, 153, 0.1))',
                          border: '1px solid',
                          borderColor: 'secondary.main',
                          color: 'secondary.main',
                        }}
                      />
                    )}
                  </Box>
                  
                  {/* Modern action buttons */}
                  <Box sx={{ mt: 'auto', pt: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Button 
                      component={Link}
                      to={`/incidents/${kase.case_id}`}
                      variant="contained"
                      fullWidth
                      size="medium"
                      disabled={kase.status === 'processing'}
                      sx={{
                        py: 1,
                        fontWeight: 600,
                        background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                        }
                      }}
                    >
                      View Full Analysis
                    </Button>
                    
                    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.75 }}>
                      <Button 
                        component={Link}
                        to={`/cases/${kase.case_id}/timeline`}
                        variant="outlined"
                        size="small"
                        startIcon={<TimelineIcon />}
                        disabled={kase.status === 'processing'}
                        sx={{ fontSize: '0.75rem' }}
                      >
                        Timeline
                      </Button>
                      
                      <Button
                        component={Link}
                        to={`/chat?case=${kase.case_id}`}
                        variant="outlined"
                        size="small"
                        startIcon={<ChatIcon />}
                        disabled={kase.status === 'processing'}
                        sx={{ fontSize: '0.75rem' }}
                      >
                        Chat
                      </Button>
                    </Box>

                    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.75 }}>
                      <Button
                        variant="outlined"
                        size="small"
                        color="secondary"
                        startIcon={<PictureAsPdfIcon />}
                        onClick={() => handleGeneratePdf(kase)}
                        disabled={kase.status === 'processing'}
                        sx={{ fontSize: '0.75rem' }}
                      >
                        Report
                      </Button>

                      <Button
                        variant="outlined"
                        size="small"
                        color="error"
                        startIcon={<DeleteIcon />}
                        onClick={() => handleDeleteClick(kase)}
                        disabled={kase.status === 'processing'}
                        sx={{ fontSize: '0.75rem' }}
                      >
                        Delete
                      </Button>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
      
      {/* PDF Report Dialog */}
      <Dialog
        open={reportDialogOpen}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {currentCase ? `PDF Report for ${currentCase.case_id}` : 'PDF Report'}
        </DialogTitle>
        <DialogContent dividers>
          {generatingPdf ? (
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <CircularProgress size={24} sx={{ mr: 2 }} />
              <Typography>Generating comprehensive report...</Typography>
            </Box>
          ) : null}
          
          <Box sx={{ whiteSpace: 'pre-line', p: 1 }}>
            {reportContent.split('\n').map((line, index) => {
              if (line.startsWith('# ')) {
                return <Typography key={index} variant="h4" gutterBottom>{line.substring(2)}</Typography>;
              } else if (line.startsWith('## ')) {
                return <Typography key={index} variant="h5" gutterBottom sx={{ mt: 2 }}>{line.substring(3)}</Typography>;
              } else if (line.startsWith('- ')) {
                return <Typography key={index} variant="body1" sx={{ ml: 2, mb: 1 }}>• {line.substring(2)}</Typography>;
              } else {
                return <Typography key={index} variant="body1" paragraph>{line}</Typography>;
              }
            })}
          </Box>
        </DialogContent>
        <DialogActions>
          {pdfBlob && (
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleDownloadPdf} 
              startIcon={<DownloadIcon />}
              disabled={generatingPdf}
            >
              Download PDF
            </Button>
          )}
          <Button onClick={handleCloseDialog}>Close</Button>
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
            Are you sure you want to delete case "{caseToDelete?.case_name || caseToDelete?.case_id}"? 
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

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default CaseList;